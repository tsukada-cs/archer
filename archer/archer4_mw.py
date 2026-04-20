#%%
import logging

import numpy as np

from archer.utilities import plot_tools
from archer.utilities import map_tools
from archer.utilities import score_funcs
from archer.utilities import conversions
from archer.utilities import nav_tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def _get_ring_weight(vmax):
    if vmax < 45:
        return 0.005 # very very small
    elif vmax >= 45 and vmax < 84:
        return 0.0694
    elif vmax >= 84:
        return 0.0263
    else:
        logger.error('`op_vmax` must be a scalar')
        return np.nan

def _get_mask_val(channel_type):
    if channel_type == '89GHz':
        return 245 # 245 avoids ice
    elif channel_type == '37GHz':
        return 50 # Basically, no mask
    elif channel_type == '183GHz':
        return 50 # Basically, no mask
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return np.nan

def _get_structure_height_km(channel_type):
    if channel_type == '89GHz':
        return 10
    elif channel_type == '37GHz':
        return 3
    elif channel_type == '183GHz':
        return 3
    elif channel_type == '183GHz':
        return 12
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return np.nan

def _get_penalty_weight(channel_type):
    if channel_type == '89GHz':
        return 1.0
    elif channel_type == '37GHz':
        return 1.0
    elif channel_type == '183GHz':
        return 1.0
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return np.nan

def archer4_mw(image, attrib, first_guess, alpha=np.deg2rad(5), para_fix=True, display_filename=None):
    """
    # archer4_mw:

    # This function performs the high-level logic for archer on microwave data
    # For complete information on the inputs and outputs, refer to archer4.py
    # AJW, CIMSS, Apr 2020.
    """
    # Dictionaries to return:
    in_dict = {}
    out_dict = {}
    score_dict = {}

    # Channel-specific settings
    ring_weight = _get_ring_weight(first_guess['vmax'])
    structure_height_km = _get_structure_height_km(attrib['archer_channel_type'])
    mask_val = _get_mask_val(attrib['archer_channel_type'])
    penalty_weight = _get_penalty_weight(attrib['archer_channel_type'])

    if attrib['archer_channel_type'] == '89GHz':
        image['bt_grid'] = map_tools.nan_around_coasts_89(image['lon_grid'], image['lat_grid'], image['data_grid'])
    elif attrib['archer_channel_type'] in ('37GHz', '183GHz'): # Not as well validated
        image['bt_grid'] = image['data_grid']

    # Revamp lon grid to prevent being cut in half in case it straddles the antemeridian
    image['lon_grid'] = nav_tools.antemeridian_decross(image['lon_grid'], first_guess['lon'])

    # Parallax fix
    if para_fix:
        if attrib['scan_type'] == 'Conical':
            image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_conical(
                image['lon_grid'], image['lat_grid'], attrib['sensor'], structure_height_km)
        elif attrib['scan_type'] == 'Crosstrack':
            image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_crosstrack(
                image['lon_grid'], image['lat_grid'], attrib['sensor'], attrib['archer_channel_type'], structure_height_km)
    else:
        logger.warning('There is no parallax fix here because archer does not recognize a valid attrib[''scan_type'']')
        image['lon_pc_grid'], image['lat_pc_grid'] = image['lon_grid'], image['lat_grid']

    # Revamp lon grid to prevent being cut in half in case it straddles the antemeridian
    # image['lon_pc_grid'] = nav_tools.antemeridian_decross(image['lon_pc_grid'])
    logger.info('Computing center-fix on whole image...')

    # Write all the input data to a dictionary
    # Note here that the naming of "default" lat/lon changes between in_dict and image. 
    # image lat/lon grid is *unaltered* lat/lon. However, in_dict lat/lon is the lat/lon
    # *to be used in Archer*.
    in_dict['sensor'] = attrib['archer_channel_type']
    in_dict['lon_mx'] = image['lon_pc_grid']
    in_dict['lat_mx'] = image['lat_pc_grid']
    in_dict['bt_mx'] = image['bt_grid']
    in_dict['time'] = first_guess['time']
    in_dict['op_lon'] = first_guess['lon']
    in_dict['op_lat'] = first_guess['lat']
    in_dict['op_vmax'] = first_guess['vmax']
    in_dict['ring_weight'] = ring_weight # Just for display purposes

    # Calculate gridded score components
    score_dict = score_funcs.combo_parts_calc_3_0(in_dict, penalty_weight=penalty_weight, alpha=alpha)

    # Clean up the spiral and ring score to allow a combo center in nan (incl. 
    # cloud-masked) areas
    np.nan_to_num(score_dict['spiral_score_grid'], nan=-1e9, copy=False)
    np.nan_to_num(score_dict['ring_score_grid'], nan=0, copy=False)

    # Calculate combo score, target point
    combo_grid = (score_dict['spiral_score_grid'] - score_dict['penalty_grid']) + ring_weight * score_dict['ring_score_grid']
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
    confidence_grid = (score_dict['spiral_score_grid'] - 0*score_dict['penalty_grid']) + ring_weight * score_dict['ring_score_grid']
    confidence_max = np.nanmax(confidence_grid)
    confidence_max_idx = np.argmax(confidence_grid)
    i_conf_max, j_conf_max = np.unravel_index(confidence_max_idx, confidence_grid.shape)

    lon_conf_max = score_dict['lon_grid1'][i_conf_max, j_conf_max]
    lat_conf_max = score_dict['lat_grid1'][i_conf_max, j_conf_max]

    # Compute distance squared without generating multiple intermediate full-size arrays
    dlat = score_dict['lat_grid1'] - lat_conf_max
    dlon = score_dict['lon_grid1'] - lon_conf_max
    cos_lat_factor = np.cos(np.deg2rad(lat_conf_max))
    dist_squared_grid = np.square(dlat)
    dist_squared_grid += np.square(cos_lat_factor * dlon)

    CONFIDENCE_DIST_DEG = 0.75
    confidence_score = confidence_max - np.nanmax(confidence_grid[dist_squared_grid > CONFIDENCE_DIST_DEG ** 2])

    alpha = conversions.confidence_to_alpha(confidence_score, attrib['archer_channel_type'], 0, in_dict['op_vmax'])

    # Represent the center fix uncertainty in terms of radius of 50%
    # confidence and radius of 95% confidence
    x_arr = np.arange(0, 10, 0.01)
    cdf_arr = 1 - (alpha * x_arr + 1) * np.exp(-alpha * x_arr)
    nearest_idx = np.argmin(np.abs(cdf_arr - 0.50))
    rad_50 = x_arr[nearest_idx]
    nearest_idx = np.argmin(np.abs(cdf_arr - 0.95))
    rad_95 = x_arr[nearest_idx]

    # Calculate the probability of having detected an eye (only for 89GHz)
    if attrib['archer_channel_type'] == '89GHz':
       eye_prob_stat = confidence_score * ring_score
       calib_stat_arr = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70,  75,  80]
       calib_perc_arr = [0, 9, 27, 44, 56, 72, 82, 90, 94, 99, 100, 100]
       eye_prob = np.interp(eye_prob_stat, calib_stat_arr, calib_perc_arr)
    else:
       eye_prob = None

    # Pack up the output values
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
        # This is an official center-fix
        out_dict['center_lon'] = nav_tools.antemeridian_restore(lon_combo_max)
        out_dict['center_lat'] = nav_tools.antemeridian_restore(lat_combo_max)
        out_dict['weak_center_lon'] = None
        out_dict['weak_center_lat'] = None
    else:
        # This is a center-fix if you must, but it's not official because it's probably corrupted
        out_dict['center_lon'] = None
        out_dict['center_lat'] = None
        out_dict['weak_center_lon'] = nav_tools.antemeridian_restore(lon_combo_max)
        out_dict['weak_center_lat'] = nav_tools.antemeridian_restore(lat_combo_max)

    # Plot test fig
    if display_filename is not None:
        import matplotlib.pyplot as plt
        fig = plot_tools.plot_diag_4panel(image, attrib, in_dict, out_dict, score_dict, display_filename=display_filename)
        plt.close(fig)
    return in_dict, out_dict, score_dict