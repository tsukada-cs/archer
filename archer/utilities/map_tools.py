#%%
import os
import logging

import numpy as np
import netCDF4

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def distance_deg(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in degrees (scalar or array)."""
    lat_dist = lat1 - lat2
    avg_lat  = (lat1 + lat2) / 2
    lon_dist = (lon1 - lon2) * np.cos(np.radians(avg_lat))
    return np.hypot(lat_dist, lon_dist)
 
def distance_deg_matrix(lat1, lon1, lat2, lon2):
    """
    Vectorised pairwise distance.
 
    Parameters
    ----------
    lat1, lon1 : (N,) ndarray  – grid points
    lat2, lon2 : (M,) ndarray  – coast points
 
    Returns
    -------
    dist : (N, M) ndarray
    """
    lat1 = lat1[:, None]
    lon1 = lon1[:, None]
    lat_dist = lat1 - lat2
    avg_lat  = (lat1 + lat2) / 2
    lon_dist = (lon1 - lon2) * np.cos(np.radians(avg_lat))
    return np.hypot(lat_dist, lon_dist)

def nan_around_coasts_89(lon_nopc_mx, lat_nopc_mx, bth_mx):
    """
    Mask swath pixels that are within one effective FOV of a coastline.
 
    Parameters
    ----------
    lon_nopc_mx : (R, C) ndarray  – pixel longitudes
    lat_nopc_mx : (R, C) ndarray  – pixel latitudes
    bth_mx      : (R, C) ndarray  – brightness temperature (or similar);
                                    pixels with value < 260 are ignored
 
    Returns
    -------
    cleared_swath : (R, C) float ndarray  – bth_mx with coastal pixels set to NaN
    """
    # ------------------------------------------------------------------
    # 1. Load coast points
    # ------------------------------------------------------------------
    this_dir  = os.path.dirname(os.path.realpath(__file__))
    nc_path   = os.path.join(this_dir, '../etc/new_world.nc')
    with netCDF4.Dataset(nc_path) as ncfile:
        coasts_lon = ncfile.variables['coasts_lon'][:].filled(np.nan)
        coasts_lat = ncfile.variables['coasts_lat'][:].filled(np.nan)
 
    valid_coast = ~np.isnan(coasts_lon) & ~np.isnan(coasts_lat)
    coasts_lon  = coasts_lon[valid_coast]
    coasts_lat  = coasts_lat[valid_coast]
 
    # ------------------------------------------------------------------
    # 2. Effective FOV (half a pixel diagonal + fudge factor)
    # ------------------------------------------------------------------
    num_rows, num_cols = bth_mx.shape
    center_col = num_cols // 2
    nadir_row_dist = distance_deg(
        lat_nopc_mx[0, center_col], lon_nopc_mx[0, center_col],
        lat_nopc_mx[1, center_col], lon_nopc_mx[1, center_col],
    )
    fov_eff = nadir_row_dist / 2 * 1.45  # diagonal + 0.05 fudge
 
    # ------------------------------------------------------------------
    # 3. Restrict coast points to swath bounding box (+fov margin)
    # ------------------------------------------------------------------
    min_lon = lon_nopc_mx.min()
    max_lon = lon_nopc_mx.max()
    min_lat = lat_nopc_mx.min()
    max_lat = lat_nopc_mx.max()
 
    in_domain = (
        (coasts_lon > min_lon) & (coasts_lon < max_lon) &
        (coasts_lat > min_lat) & (coasts_lat < max_lat)
    )
    coasts_lon = coasts_lon[in_domain]
    coasts_lat = coasts_lat[in_domain]
 
    # ------------------------------------------------------------------
    # 4. Build swath mask
    # ------------------------------------------------------------------
    swath_mask = np.zeros(bth_mx.shape, dtype=bool)
 
    if coasts_lon.size == 0:
        # No coast points in domain – nothing to mask
        cleared_swath = bth_mx.astype(float)
        logger.info(f'Total points masked for coastal boundaries: {0}')
        return cleared_swath
 
    # Candidate pixels (bth >= 260)
    valid_rows, valid_cols = np.where(bth_mx >= 260)
    if valid_rows.size == 0:
        cleared_swath = bth_mx.astype(float)
        logger.info(f'Total points masked for coastal boundaries: {0}')
        return cleared_swath
 
    lat_v = lat_nopc_mx[valid_rows, valid_cols] # (N,)
    lon_v = lon_nopc_mx[valid_rows, valid_cols]
 
    # Use cKDTree for fast spatial queries instead of O(N*M) distance matrix
    from scipy.spatial import cKDTree
    tree_pts = cKDTree(np.column_stack((lat_v, lon_v)))
    tree_coast = cKDTree(np.column_stack((coasts_lat, coasts_lon)))
    
    # 1.6 accounts for max cos(lat) stretch in the distance formula
    pairs = tree_pts.query_ball_tree(tree_coast, r=1.6 * fov_eff)
    
    i_list = []
    j_list = []
    for i, j_idx in enumerate(pairs):
        if j_idx:
            i_list.extend([i] * len(j_idx))
            j_list.extend(j_idx)
            
    if i_list:
        lat1 = lat_v[i_list]
        lon1 = lon_v[i_list]
        lat2 = coasts_lat[j_list]
        lon2 = coasts_lon[j_list]
        
        exact_dist = distance_deg(lat1, lon1, lat2, lon2)
        valid_mask = exact_dist < fov_eff
        
        swath_flags = np.zeros(len(lat_v), dtype=bool)
        swath_flags[np.array(i_list)[valid_mask]] = True
        swath_mask[valid_rows[swath_flags], valid_cols[swath_flags]] = True
 
    cleared_swath = bth_mx.astype(float)
    cleared_swath[swath_mask] = np.nan
 
    logger.info(f'Total points masked for coastal boundaries: {int(swath_mask.sum())}')
    return cleared_swath