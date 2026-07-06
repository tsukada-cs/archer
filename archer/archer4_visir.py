#%%
import logging

import numpy as np
import scipy.ndimage as ndi

from archer.utilities import plot_tools
from archer.utilities import score_funcs
from archer.utilities import conversions
from archer.utilities import nav_tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def dilate_nan(bt_mx, dilate_val=2):
    """Expand NaN values to surrounding area"""
    bt_mx = bt_mx.copy()
    
    # Create a mask of NaN values
    nan_mask = np.isnan(bt_mx)
    
    # Create a structuring element for dilation
    struct = np.ones((2*dilate_val + 1, 2*dilate_val + 1), dtype=bool)
    dilated_mask = ndi.binary_dilation(nan_mask, structure=struct)
    
    # Fill the dilated mask with NaN
    bt_mx[dilated_mask] = np.nan
    return bt_mx

def archer4_visir(image, attrib, first_guess, alpha=np.deg2rad(5), para_fix=True, display_filename=None):
    # archer4_visir:
    # This function performs the high-level logic for archer on infrared and visible data.
    # For complete information on the inputs and outputs, refer to archer4.py
    # AJW, CIMSS, Apr 2020.

    # Dictionaries to return:
    in_dict = {}
    out_dict = {}
    score_dict = {}

    # Channel-specific settings
    if attrib['archer_channel_type'] == 'IR':
        structure_height_km = 16
        mask_val = 265 # 265: Keep low cloud only
        ring_weight = 0.0167
        penalty_weight = 0.33
        image['bt_grid'] = image['data_grid'] # Archer acts on BT
    elif attrib['archer_channel_type'] == 'SWIR':
        structure_height_km = 16
        mask_val = -1e6 # 265: Keep low cloud only
        ring_weight = 0.0167
        penalty_weight = 0.33
        image['bt_grid'] = image['data_grid'] # Archer acts on BT
    elif attrib['archer_channel_type'] in ['Vis', 'DNB']:
        structure_height_km = 2
        mask_val = -1e6
        ring_weight = 0.0020
        penalty_weight = 0.33

        if attrib['scan_type'] == 'Geo' and attrib['archer_channel_type'] == 'Vis':
            # Normalize for solar zenith angle. This matters because a difference
            # in average brightness value from image to image does indeed throw off
            # the results.
            cos_solar_zenith_grid = nav_tools.cos_solar_zenith(
                image['lon_grid'], image['lat_grid'], first_guess['time'])

            # Sun must be >1.5 degrees above horizon for that conversion to make sense
            cos_solar_zenith_grid[cos_solar_zenith_grid < 0.0262] = np.NaN

            # Convert to pseudo brightness temp for Archer to act on using optimized sqrt
            bv_norm = image['data_grid'] / np.sqrt(cos_solar_zenith_grid)
            image['bt_grid'] = 350 - 0.75 * bv_norm

            # Filter out nighttime cases
            num_pix = np.prod(np.shape(image['bt_grid']))
            num_nan = np.sum(np.isnan(image['bt_grid']), axis=(0,1))
            if num_nan / num_pix > 0.3:
                logger.warning('Too dark for ARCHER. Exiting.')
                return in_dict, out_dict, score_dict
        else:
            # Assume polar Vis/DNB imagery is already normalized:
            image['bt_grid'] = 350 - 0.75*image['data_grid']

    # Revamp lon grid to prevent being cut in half in case it straddles the antemeridian
    image['lon_grid'] = nav_tools.antemeridian_decross(image['lon_grid'], first_guess['lon'])

    # Parallax fix
    if para_fix:
        if 'zen_grid' in image.keys() and 'azm_grid' in image.keys():
            image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_allnav(
                image['lon_grid'], image['lat_grid'], 
                image['zen_grid'], image['azm_grid'], structure_height_km)
        else:
            if attrib['scan_type'] == 'Geo':
                image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_geo(
                    image['lon_grid'], image['lat_grid'], 
                    attrib['nadir_lon'], attrib['sensor'], structure_height_km)
            elif attrib['scan_type'] == 'Crosstrack':
                image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_crosstrack(
                    image['lon_grid'], image['lat_grid'], 
                    attrib['sensor'], attrib['archer_channel_type'], 
                    structure_height_km)
            else: 
                logger.error('attrib[''scan_type''] is not valid and cannot parallax fix. Exiting.')
                return in_dict, out_dict, score_dict
    else:
        image['lon_pc_grid'], image['lat_pc_grid'] = \
            image['lon_grid'], image['lat_grid']

    # Reduce resolution of the image data to prevent lags and artifacts
    image = nav_tools.reduce_res(image, step_km=4)

    # Write all the input data to a dictionary
    in_dict['sensor'] = attrib['archer_channel_type']
    in_dict['op_lon'] = first_guess['lon']
    in_dict['op_lat'] = first_guess['lat']
    in_dict['op_vmax'] = first_guess['vmax']
    in_dict['time'] = first_guess['time']
    in_dict['ring_weight'] = ring_weight # Just for display purposes


    # Iterate through feature-level and then surface-level center-fixes and keep the best:
    # (BTW, I am not proud of this code construction. It stems from something that was 
    # not very straightforward in the original Matlab syntax.)
    confidence_score_best = -1e6 # Start with this value for upcoming comparisons

    for level in ['feature', 'surface']:
        # Note here that the naming of "default" lat/lon changes between in_dict and image. 
        # image lat/lon grid is *unaltered* lat/lon. However, in_dict lat/lon is the lat/lon
        # *to be used in Archer*.
        if level == 'feature':
            logger.info('Computing center-fix on full image...')
            in_dict['lon_mx'] = image['lon_pc_grid']
            in_dict['lat_mx'] = image['lat_pc_grid']
            in_dict['bt_mx'] = image['bt_grid']
        elif level == 'surface':
            logger.info('Computing center-fix on cloud-masked image (w/o parallax fix) ...')
            in_dict['lon_mx'] = image['lon_grid']
            in_dict['lat_mx'] = image['lat_grid']
            in_dict['bt_mx'][in_dict['bt_mx'] < mask_val] = np.NaN

            # With IR imagery, we have to consider pixels with partial clouds/convection.
            # Dilate these features by 2 (NOTE: for ~10 km resolution):
            if attrib['archer_channel_type'] == 'IR':
                dilate_val = 2
                in_dict['bt_mx'] = dilate_nan(in_dict['bt_mx'], dilate_val)

        # Calculate gridded score components
        score_dict = score_funcs.combo_parts_calc_3_0(in_dict, penalty_weight=penalty_weight, alpha=alpha)

        # Clean up the spiral and ring score to allow a combo center in nan (incl. 
        # cloud-masked) areas
        np.nan_to_num(score_dict['spiral_score_grid'], nan=-1e9, copy=False)
        np.nan_to_num(score_dict['ring_score_grid'], nan=0, copy=False)

        # Calculate combo score, target point
        combo_grid = (score_dict['spiral_score_grid'] - score_dict['penalty_grid']) + \
            ring_weight * score_dict['ring_score_grid']
        score_dict['combo_score_grid'] = combo_grid

        combo_score = np.max(combo_grid)
        max_idx = np.argmax(combo_grid)
        i_combo_max, j_combo_max = np.unravel_index(max_idx, combo_grid.shape)

        lon_combo_max = score_dict['lon_grid1'][i_combo_max, j_combo_max]
        lat_combo_max = score_dict['lat_grid1'][i_combo_max, j_combo_max]

        ring_radius_deg = score_dict['ring_radius_grid'][i_combo_max, j_combo_max]
        ring_score = score_dict['ring_score_grid'][i_combo_max, j_combo_max]
        score_by_radius_arr = np.squeeze(score_dict['ring_score_grid_full'][i_combo_max, j_combo_max, :])
        gradient_grid = np.squeeze(score_dict['radial_gradient_4d'][i_combo_max, j_combo_max, :])

        # Calculate confidence score using combo grid w/o the distance penalty.
        # The confidence score is the maximum value minus the highest score CONFIDENCE_DIST_DEG away
        confidence_grid = (score_dict['spiral_score_grid'] - 0*score_dict['penalty_grid']) + \
            ring_weight * score_dict['ring_score_grid']
        confidence_max = np.nanmax(confidence_grid)
        confidence_max_idx = np.argmax(confidence_grid)
        i_conf_max, j_conf_max = np.unravel_index(confidence_max_idx, confidence_grid.shape)

        lon_conf_max = score_dict['lon_grid1'][i_conf_max, j_conf_max]
        lat_conf_max = score_dict['lat_grid1'][i_conf_max, j_conf_max]

        # Compute distance squared mapping with less memory overhead
        dlat = score_dict['lat_grid1'] - lat_conf_max
        dlon = score_dict['lon_grid1'] - lon_conf_max
        cos_lat_factor = np.cos(np.deg2rad(lat_conf_max))
        dist_squared_grid = np.square(dlat)
        dist_squared_grid += np.square(cos_lat_factor * dlon)

        CONFIDENCE_DIST_DEG = 0.75
        confidence_score = confidence_max - \
            np.nanmax(confidence_grid[dist_squared_grid > CONFIDENCE_DIST_DEG ** 2])

        alpha = conversions.confidence_to_alpha(
            confidence_score, attrib['archer_channel_type'], 0, in_dict['op_vmax'])

        # Represent the center fix uncertainty in terms of radius of 50%
        # confidence and radius of 95% confidence
        x_arr = np.arange(0, 10, 0.01)
        cdf_arr = 1 - (alpha * x_arr +1) * np.exp(-alpha * x_arr)
        nearest_idx = np.argmin(np.abs(cdf_arr - 0.50))
        rad_50 = x_arr[nearest_idx]
        nearest_idx = np.argmin(np.abs(cdf_arr - 0.95))
        rad_95 = x_arr[nearest_idx]

        # Calculate the probability of having detected an eye (only works for IR)
        if attrib['archer_channel_type'] == 'IR':
           eye_prob_stat = confidence_score * ring_score
           calib_stat_arr = [0, .1, .5,  1, 1.5,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  13,  14,  15]
           calib_perc_arr = [0,  1,  7, 15,  40, 47, 50, 60, 65, 70, 75, 80, 85, 95, 98, 99, 100, 100, 100]
           eye_prob = np.interp(eye_prob_stat, calib_stat_arr, calib_perc_arr)
        else:
           eye_prob = None

        uses_target = score_funcs.quality_check(score_dict)
        out_dict['uses_target'] = uses_target
        out_dict['archer_channel_type'] = attrib['archer_channel_type']
        out_dict['ring_radius_deg'] = ring_radius_deg
        out_dict['score_by_radius_arr'] = score_by_radius_arr
        out_dict['confidence_score'] = confidence_score
        out_dict['alpha_parameter'] = alpha
        out_dict['radius50percCertDeg'] = rad_50
        out_dict['radius95percCertDeg'] = rad_95
        out_dict['eye_prob'] = eye_prob

        if uses_target:
            out_dict['center_lon'] = lon_combo_max
            out_dict['center_lat'] = lat_combo_max
            out_dict['weak_center_lon'] = None
            out_dict['weak_center_lat'] = None

            if level == 'feature':
                # If it made it this far successfully, then no need to try the next option ('surface')
                break 
        else:
            out_dict['center_lon'] = None
            out_dict['center_lat'] = None
            out_dict['weak_center_lon'] = lon_combo_max
            out_dict['weak_center_lat'] = lat_combo_max

        # If continuing to the next step in the loop, carry on with the new best conf score
        confidence_score_best = np.max([confidence_score, confidence_score_best])

    # Plot test fig
    if display_filename is not None:
        plot_tools.plot_diag_4panel(image, attrib, in_dict, out_dict, score_dict, display_filename=display_filename)

    return in_dict, out_dict, score_dict

