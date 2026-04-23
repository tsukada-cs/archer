#%%
import logging

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')

def distance_deg(lon1, lat1, lon2, lat2):
    lat_dist_deg = lat1 - lat2
    avg_lat = np.mean(lat2)
    lon_dist_deg = (lon1 - lon2) * np.cos(np.deg2rad(avg_lat))
    return np.hypot(lat_dist_deg, lon_dist_deg)

def combo_parts_calc_3_0(
    mi_dict, 
    penalty_weight=1.0, 
    mask_val=None,
    alpha=np.deg2rad(5), 
    gradient_mode="sobel",
):
    """
    Calculate the spiral and ring scores for a given microwave imager dictionary.

    Parameters
    ----------
    mi_dict : dict
        Dictionary containing the microwave imager data.
    penalty_weight : float
        Weighting factor for the penalty term.
    alpha : float
        Angle of the spiral center in radians.

    Returns
    -------
    combo_score_dict : dict
        Dictionary containing the spiral and ring scores.
    """
    # Resampling, spiral and ring parameters
    if mi_dict['sensor'].lower() in ('89ghz', '37ghz', '183ghz'):
        LON_INC = 0.05
        LAT_INC = 0.05
        PERIM_DEG = 1.6
        FILTER_RADIUS_DEG = 2.0
        SPIRAL_SEARCH_RADIUS_DEG = 2.0
        MAX_RADIUS_DEG = 2.50
    else:
        LON_INC = 0.025
        LAT_INC = 0.025
        PERIM_DEG = 2.5
        FILTER_RADIUS_DEG = 2.5
        SPIRAL_SEARCH_RADIUS_DEG = 2.0
        MAX_RADIUS_DEG = 0.50

    # Universal spiral parameters
    SPIRAL_WEIGHT = 15
    SPIRAL_OFFSET = 20
    SPRIAL_SPACING_DEG = 0.05

    # Universal ring parameters
    RING_WEIGHT = 250.0
    MIN_RADIUS_DEG = 0.05

    # Remove any false data
    if mi_dict['sensor'].lower() in ('89ghz', '37ghz', '183ghz', 'ir'):
        with np.errstate(invalid='ignore'):
            mi_dict['bt_mx'][(mi_dict['bt_mx'] < 80)] = np.nan

    # Resample the swath to a regular grid, centred on the fx point, and with 
    # the correct aspect ratio at the fx point
    x_arr_offset_gcd = np.arange(-PERIM_DEG, PERIM_DEG+1e-6, LON_INC)
    y_arr_offset_gcd = np.arange(PERIM_DEG, -PERIM_DEG-1e-6, -LAT_INC)
    x_grid_offset_gcd, y_grid_offset_gcd = np.meshgrid(x_arr_offset_gcd, y_arr_offset_gcd)
    lat_arr1 = y_arr_offset_gcd + mi_dict['op_lat']
    lat_grid1 = np.tile(lat_arr1[:, np.newaxis], (1, len(x_arr_offset_gcd)))
    lon_grid1 = x_grid_offset_gcd / np.cos(np.deg2rad(lat_grid1)) + mi_dict['op_lon']

    x_flat = mi_dict['lon_mx'].ravel()
    y_flat = mi_dict['lat_mx'].ravel()
    b_flat = mi_dict['bt_mx'].ravel()
    valid = ~(np.isnan(x_flat) | np.isnan(y_flat) | np.isnan(b_flat))
    
    if np.any(valid):
        tree = cKDTree(np.column_stack((x_flat[valid], y_flat[valid])))
        query_pts = np.column_stack((lon_grid1.ravel(), lat_grid1.ravel()))
        max_dist = 1.5 * LON_INC
        dists, idxs = tree.query(query_pts, k=3, distance_upper_bound=max_dist)
        data_grid1_flat = np.full(query_pts.shape[0], np.nan)
        valid_queries = dists[:, 0] < np.inf
        if np.any(valid_queries):
            b_valid = b_flat[valid]
            v_dists = dists[valid_queries]
            v_idxs = idxs[valid_queries]
            eps = 1e-12
            weights = 1.0 / (v_dists**2 + eps)

            valid_neighbors = v_dists < np.inf
            weights[~valid_neighbors] = 0.0

            safe_idxs = np.where(valid_neighbors, v_idxs, 0)
            vals = b_valid[safe_idxs]
            
            data_grid1_flat[valid_queries] = np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)
        data_grid1 = data_grid1_flat.reshape(lon_grid1.shape)
    else:
        logger.error('No valid points to interpolate')
        data_grid1 = np.full(np.shape(lon_grid1), np.nan)

    # Spiral center
    spiral_center_calc_args = {
        'x_grid_offset_gcd': x_grid_offset_gcd,
        'y_grid_offset_gcd': y_grid_offset_gcd,
        'data_grid1': data_grid1,
        'sensor_type': mi_dict['sensor'],
        'op_lon': mi_dict['op_lon'],
        'op_lat': mi_dict['op_lat'],
        'filter_radius_deg': FILTER_RADIUS_DEG,
        'spiral_search_radius_deg': SPIRAL_SEARCH_RADIUS_DEG,
        'spiral_spacing_deg': SPRIAL_SPACING_DEG,
        'alpha': alpha,
        'mask_val': None,
        'gradient_mode': gradient_mode,
    }
    sp_grid1, fraction_input = spiral_center_calc(**spiral_center_calc_args)
    spiral_score_grid = SPIRAL_WEIGHT * sp_grid1 - SPIRAL_OFFSET

    # Add in the penalty for distance from first guess
    penalty_grid = penalty_weight * distance_deg(mi_dict['op_lon'], mi_dict['op_lat'], lon_grid1, lat_grid1)
    spiral_score_grid_with_penalty = spiral_score_grid - penalty_grid

    # Find the swarm of points to test for ring fitting
    SPIRAL_FIT_BUFFER = 1.5 # Include all points within this range of the max value ...
    SWARM_REACH = 0.25 # ... plus this distance out

    spiral_score_grid_with_penalty_nonan = spiral_score_grid_with_penalty
    spiral_score_grid_with_penalty_nonan[np.isnan(spiral_score_grid_with_penalty)] = -1e9
    is_inside_buffer = spiral_score_grid_with_penalty_nonan > np.nanmax(spiral_score_grid_with_penalty) - SPIRAL_FIT_BUFFER

    # Expand the buffer by SWARM_REACH
    valid_lons = lon_grid1[is_inside_buffer]
    valid_lats = lat_grid1[is_inside_buffer]
    
    if len(valid_lats) > 0:
        # Utilize fast morphological binary dilation to replace O(N*M) distance calculations
        import scipy.ndimage as ndi
        
        # Determine the physical degree increment from the grid
        lat_inc_val = np.abs(lat_grid1[0, 0] - lat_grid1[1, 0])
        if lat_inc_val == 0:
            lat_inc_val = 0.05
            
        swarm_reach_pixels = int(np.ceil(SWARM_REACH / lat_inc_val))
        
        # Create a circular structural element corresponding to the physical SWARM_REACH
        y_str, x_str = np.ogrid[-swarm_reach_pixels:swarm_reach_pixels+1, -swarm_reach_pixels:swarm_reach_pixels+1]
        struct = x_str**2 + y_str**2 <= (SWARM_REACH / lat_inc_val)**2
        
        # Perform binary dilation on the mask directly
        is_in_bounds = ndi.binary_dilation(is_inside_buffer, structure=struct)
    else:
        is_in_bounds = np.zeros(np.shape(lat_grid1), dtype=bool)

    # Ring center
    ring_score_dict = ring_score_calc(
        x_grid_offset_gcd, 
        y_grid_offset_gcd, 
        data_grid1,
        mi_dict['sensor'], 
        is_in_bounds, 
        MIN_RADIUS_DEG, 
        MAX_RADIUS_DEG,
        mask_val=mask_val,
        gradient_mode=gradient_mode
    )
    
    # Create score dictionary
    combo_score_dict = {}
    combo_score_dict['lon_grid1'] = lon_grid1
    combo_score_dict['lat_grid1'] = lat_grid1
    combo_score_dict['data_grid1'] = data_grid1
    combo_score_dict['spiral_score_grid'] = spiral_score_grid
    combo_score_dict['penalty_grid'] = penalty_grid
    combo_score_dict['ring_score_grid'] = RING_WEIGHT * ring_score_dict['ring_score_grid']
    combo_score_dict['ring_radius_grid'] = ring_score_dict['ring_radius_grid']
    combo_score_dict['ring_score_grid_full'] = ring_score_dict['ring_score_grid_full']
    combo_score_dict['radial_gradient_4d'] = ring_score_dict['radial_gradient_4d']
    combo_score_dict['fraction_input'] = fraction_input
    return combo_score_dict

def spiral_center_calc(
    x_grid_offset_gcd, 
    y_grid_offset_gcd, 
    data_grid1, 
    sensor_type,
    op_lon, 
    op_lat, 
    filter_radius_deg, 
    spiral_search_radius_deg, 
    spiral_spacing_deg,
    mask_val=None,
    alpha=np.deg2rad(5),
    gradient_mode="sobel"
):
    # Sensor-specfic settings
    if sensor_type.lower() == '37ghz':
        outside_factor = 0.62
        data_grid1 = -data_grid1
    elif sensor_type.lower() in ('ir', 'vis'):
        outside_factor = 0.50
    else:
        outside_factor = 0.62

    # Cut down to a usable disk, surrounded by nans
    in_filter_disk = x_grid_offset_gcd**2 + y_grid_offset_gcd**2 <= filter_radius_deg**2
    if mask_val is not None:
        in_filter_disk = in_filter_disk & (data_grid1 >= mask_val)
    disk_img = np.full_like(data_grid1, np.nan)
    disk_img[in_filter_disk] = data_grid1[in_filter_disk]

    # Make 1D arrays of just the clean points. "clean" means no nans
    is_clean = ~np.isnan(disk_img)
    disk_img_clean = disk_img[is_clean]
    x_grid_offset_gcd_clean = x_grid_offset_gcd[is_clean]
    y_grid_offset_gcd_clean = y_grid_offset_gcd[is_clean]
    lon_inc = x_grid_offset_gcd[0,1] - x_grid_offset_gcd[0,0]
    lat_inc = y_grid_offset_gcd[0,0] - y_grid_offset_gcd[1,0]

    if gradient_mode.lower() == "sobel":
        grad_e = ndi.sobel(disk_img, axis=1) / lon_inc
        grad_n = ndi.sobel(disk_img, axis=0) / lat_inc
    else:
        grad_n, grad_e = np.gradient(disk_img, lon_inc, lat_inc)
    grad_n = -grad_n
    grad_n_clean = grad_n[is_clean]
    grad_e_clean = grad_e[is_clean]
    grad_orig_mag_clean = np.hypot(grad_n_clean, grad_e_clean)
    grad_log_mag_clean = np.log(1 + grad_orig_mag_clean)
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_log_reduction_clean = grad_log_mag_clean / grad_orig_mag_clean
    grad_n_log_clean = grad_log_reduction_clean * grad_n_clean
    grad_e_log_clean = grad_log_reduction_clean * grad_e_clean

    # 1. Iterate the cross product score on a coarse grid
    off_arr = np.arange(-spiral_search_radius_deg, spiral_search_radius_deg+1e-6, spiral_spacing_deg)

    all_center_mean_cross = np.full((len(off_arr), len(off_arr)), np.nan)
    all_center_xs, all_center_ys = np.meshgrid(off_arr, off_arr, indexing='ij')

    # Search out (search_radius_deg) degrees from the center point
    valid_mask = all_center_xs**2 + all_center_ys**2 <= (spiral_search_radius_deg + 2*spiral_spacing_deg/3)**2
    valid_x_off = all_center_xs[valid_mask]
    valid_y_off = all_center_ys[valid_mask]
    
    CHUNK_SIZE = 100
    valid_scores = np.zeros(len(valid_x_off))
    
    # Pre-calculate constant scalar values outside the loop to optimize performance
    sign_lat = np.sign(op_lat)
    sqrt_alpha_factor = np.sqrt(1 + alpha**2)
    
    for i in range(0, len(valid_x_off), CHUNK_SIZE):
        x_off_chunk = valid_x_off[i:i+CHUNK_SIZE, np.newaxis]
        y_off_chunk = valid_y_off[i:i+CHUNK_SIZE, np.newaxis]

        proxy_x_clean = x_grid_offset_gcd_clean[np.newaxis, :] - x_off_chunk
        proxy_y_clean = y_grid_offset_gcd_clean[np.newaxis, :] - y_off_chunk

        # Utilize C-level np.hypot for hypotenuse and apply pre-calculated alpha factor
        proxy_norm = np.hypot(proxy_x_clean, proxy_y_clean) * sqrt_alpha_factor

        spiral_x_clean = (alpha * proxy_x_clean + sign_lat * proxy_y_clean) / proxy_norm
        spiral_y_clean = (alpha * proxy_y_clean - sign_lat * proxy_x_clean) / proxy_norm

        raw_cross_score = spiral_x_clean * grad_n_log_clean[np.newaxis, :] - spiral_y_clean * grad_e_log_clean[np.newaxis, :]

        cross_score_clean = np.maximum(0, -raw_cross_score) + outside_factor * np.maximum(0, raw_cross_score)

        is_nan_cross = np.isnan(raw_cross_score)
        cross_score_clean[is_nan_cross] = np.nan
        valid_scores[i:i+CHUNK_SIZE] = np.nanmean(cross_score_clean, axis=1)

    all_center_mean_cross[valid_mask] = valid_scores

    # 2. Search for the best full-resolution grid cell by (cubic?) interpolation
    interp_func = RegularGridInterpolator(
        (off_arr, off_arr), all_center_mean_cross, 
        method='linear', bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack((x_grid_offset_gcd.ravel(), y_grid_offset_gcd.ravel()))
    sp_grid = interp_func(pts).reshape(x_grid_offset_gcd.shape)

    # Clean out the dodgy edge values
    sp_grid[x_grid_offset_gcd**2 + y_grid_offset_gcd**2 >= spiral_search_radius_deg**2] = np.nan
    fraction_input = disk_img_clean.size / np.sum(in_filter_disk)
    return sp_grid, fraction_input

def ring_score_calc(
    x_grid_offset_gcd, 
    y_grid_offset_gcd, 
    data_grid1, 
    sensor_type, 
    is_in_bounds, 
    min_radius_deg, 
    max_radius_deg,
    mask_val=None,
    gradient_mode="sobel"
):
    # Parameters
    da = 5 # deg
    ang_deg_arr = np.arange(0, 360, da)
    na = len(ang_deg_arr)
    ring_point_thresh = 0.425 * na

    # Add reversal step for 37GHz b/c eyes are *colder*:
    if '37' in sensor_type:
        data_grid1 = 450 - data_grid1

    # Calculate the modified gradient field
    # Somehow, the step of *1.14* is better than 1 for the *final* result. Couldn't figure out why.

    POWER = 0.333
    GRADIENT_STEP = 1.14
    if gradient_mode.lower() == "sobel":
        grad_e = ndi.sobel(data_grid1**POWER, axis=1) / GRADIENT_STEP
        grad_n = ndi.sobel(data_grid1**POWER, axis=0) / (-GRADIENT_STEP)
    else:
        grad_n, grad_e = np.gradient((data_grid1**POWER), -GRADIENT_STEP, GRADIENT_STEP) # (This has to reverse the Matlab formula)

    # Translate degrees to pixels
    deg_per_pix = np.abs(y_grid_offset_gcd[0,0] - y_grid_offset_gcd[1,0])

    # Initialize variables related to the score grids
    n_rows, n_cols = np.shape(data_grid1)
    ring_score_grid = np.full(np.shape(data_grid1), np.nan)
    ring_radius_grid = np.zeros(np.shape(data_grid1))
    max_eye_bt_grid = np.full(np.shape(data_grid1), np.nan)

    # Unit vectors pointed radially inward
    ring_unit_vector_x = -np.cos(np.deg2rad(ang_deg_arr))
    ring_unit_vector_y = -np.sin(np.deg2rad(ang_deg_arr))

    # Create offset grids 
    mid_row = np.round(n_rows/2) # =32
    mid_col = np.round(n_cols/2)
    off_col_grid, off_row_grid = np.meshgrid(range(0, n_cols) - mid_col, range(0, n_rows) - mid_row) # -32...32

    # Convert offset grids to offset arrays
    valid_mask = x_grid_offset_gcd**2 + y_grid_offset_gcd**2 < (max_radius_deg + deg_per_pix)**2
    if mask_val is not None:
        valid_mask = valid_mask & (data_grid1 >= mask_val)
    off_col_pts = off_col_grid[valid_mask]
    off_row_pts = off_row_grid[valid_mask]
    off_x_pts = x_grid_offset_gcd[valid_mask]
    off_y_pts = y_grid_offset_gcd[valid_mask]

    # Build the score grids. Iterate by radius, and within that, iterate by location
    rax = np.arange(min_radius_deg, max_radius_deg+1e-6, 0.05)

    # Extra part for ERC calcs
    ring_score_grid_full = np.full((n_rows, n_cols, rax.size), np.nan)
    radial_gradient_4d = np.full((n_rows, n_cols, rax.size, na), np.nan)

    for rad_idx in reversed(range(rax.size)):
        radius_deg = rax[rad_idx]

        # Calculate the row/col offsets for any center point at this radius
        ring_x_pts = radius_deg * np.cos(np.deg2rad(ang_deg_arr))
        ring_y_pts = radius_deg * np.sin(np.deg2rad(ang_deg_arr))
        
        dist_sq = (off_x_pts[:, np.newaxis] - ring_x_pts[np.newaxis, :])**2 + \
                  (off_y_pts[:, np.newaxis] - ring_y_pts[np.newaxis, :])**2
        nearest_idxs = np.argmin(dist_sq, axis=0)
        row_off_arr = off_row_pts[nearest_idxs]
        col_off_arr = off_col_pts[nearest_idxs]

        radius_factor = radius_deg ** 0.1

        # Vectorized iteration by location
        valid_i, valid_j = np.nonzero(is_in_bounds)
        if len(valid_i) > 0:
            ring_rows = (valid_i[:, np.newaxis] + row_off_arr[np.newaxis, :]).astype(int)
            ring_cols = (valid_j[:, np.newaxis] + col_off_arr[np.newaxis, :]).astype(int)
            
            is_in_image_box = (ring_rows >= 0) & (ring_rows < n_rows) & \
                              (ring_cols >= 0) & (ring_cols < n_cols)
                              
            ring_grad_x = np.full(ring_cols.shape, np.nan)
            ring_grad_y = np.full(ring_rows.shape, np.nan)
            
            # Efficient assignment with valid coordinates
            valid_coords = is_in_image_box
            ring_grad_x[valid_coords] = grad_e[ring_rows[valid_coords], ring_cols[valid_coords]]
            ring_grad_y[valid_coords] = grad_n[ring_rows[valid_coords], ring_cols[valid_coords]]
            
            dot_products = ring_unit_vector_x[np.newaxis, :] * ring_grad_x + \
                           ring_unit_vector_y[np.newaxis, :] * ring_grad_y
                           
            with np.errstate(invalid='ignore'):
                mean_dots = np.nanmean(dot_products, axis=1)
                
            n_reals = np.sum(~np.isnan(dot_products), axis=1)
            
            ring_scores = radius_factor * mean_dots
            ring_scores[n_reals <= ring_point_thresh] = 0.0
            ring_scores[np.isnan(ring_scores)] = 0.0
            
            if rad_idx == len(rax) - 1:
                update_mask = np.ones(len(valid_i), dtype=bool)
            else:
                update_mask = ring_scores > ring_score_grid[valid_i, valid_j]
                
            update_i = valid_i[update_mask]
            update_j = valid_j[update_mask]
            ring_score_grid[update_i, update_j] = ring_scores[update_mask]
            ring_radius_grid[update_i, update_j] = radius_deg
            
            ring_score_grid_full[valid_i, valid_j, rad_idx] = ring_scores
            radial_gradient_4d[valid_i, valid_j, rad_idx, :] = dot_products

    # Assinging the warmest pixel corresponding to each ring
    valid_i, valid_j = np.nonzero(is_in_bounds)
    if len(valid_i) > 0:
        x_cents = x_grid_offset_gcd[valid_i, valid_j]
        y_cents = y_grid_offset_gcd[valid_i, valid_j]
        radii = ring_radius_grid[valid_i, valid_j]
        
        x_flat = x_grid_offset_gcd.ravel()
        y_flat = y_grid_offset_gcd.ravel()
        data_flat = data_grid1.ravel()
        
        CHUNK_SIZE = 500
        for idx in range(0, len(valid_i), CHUNK_SIZE):
            i_chk = valid_i[idx:idx+CHUNK_SIZE]
            j_chk = valid_j[idx:idx+CHUNK_SIZE]
            r_chk = radii[idx:idx+CHUNK_SIZE, np.newaxis]
            x_chk = x_cents[idx:idx+CHUNK_SIZE, np.newaxis]
            y_chk = y_cents[idx:idx+CHUNK_SIZE, np.newaxis]

            dist_sq = (x_flat[np.newaxis, :] - x_chk)**2 + (y_flat[np.newaxis, :] - y_chk)**2
            mask = dist_sq <= r_chk**2

            for k in range(len(i_chk)):
                if np.any(mask[k]):
                    max_eye_bt_grid[i_chk[k], j_chk[k]] = np.nanmax(data_flat[mask[k]])
                else:
                    max_eye_bt_grid[i_chk[k], j_chk[k]] = np.nan

    # Put output into dictionary
    ring_score_dict = {}
    ring_score_dict['ring_score_grid'] = ring_score_grid
    ring_score_dict['ring_radius_grid'] = ring_radius_grid
    ring_score_dict['max_eye_bt_grid'] = max_eye_bt_grid
    ring_score_dict['ring_score_grid_full'] = ring_score_grid_full
    ring_score_dict['radial_gradient_4d'] = radial_gradient_4d
    return ring_score_dict

def quality_check(spiral_score_grid, fraction_input, coverage_threshold=0.5):
    """
    Check if the best center is reliable.

    Returns
    -------
    uses_target : bool
        True if the best center is reliable, False otherwise.
    """
    if fraction_input < coverage_threshold:
        return False

    i_max_score, j_max_score = np.unravel_index(
        np.nanargmax(spiral_score_grid), 
        spiral_score_grid.shape
    )
    if i_max_score <= 1 or j_max_score <= 1:
        return False
    if i_max_score >= spiral_score_grid.shape[0]-2 or j_max_score >= spiral_score_grid.shape[1]-2:
        return False
    if np.isnan(spiral_score_grid[i_max_score-2:i_max_score+3, j_max_score-2:j_max_score+3]).any():
        return False
    return True