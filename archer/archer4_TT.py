#%%
import logging
from typing import Optional

import numpy as np
import scipy.ndimage as ndi

from archer.utilities import plot_tools
from archer.utilities import map_tools
from archer.utilities import score_funcs
from archer.utilities import conversions
from archer.utilities import nav_tools
from archer.utilities import fdeck_tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def _dilate_nan(bt_mx, dilate_val=2):
    """Expand nan values to surrounding area"""
    struct = np.full((2*dilate_val+1, 2*dilate_val+1), True)
    dilated_mask = ndi.binary_dilation(np.isnan(bt_mx), structure=struct)
    return np.where(dilated_mask, np.nan, bt_mx)

def _get_ring_weight(channel_type, vmax):
    """Get ring weight for a given channel type"""
    if channel_type.lower() in ('ir', 'swir', 'vis', 'dnb'):
        return _get_ring_weight_visir(channel_type)
    elif channel_type.lower() in ('89ghz', '37ghz', '183ghz'):
        return _get_ring_weight_mw(vmax)
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return 0.0

def _get_ring_weight_visir(channel_type):
    """Get ring weight for a given channel type"""
    if channel_type.lower() in ('ir', 'swir'):
        return 0.0167
    elif channel_type.lower() in ('vis', 'dnb'):
        return 0.0020
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return 0.0

def _get_ring_weight_mw(vmax):
    """Legacy function to get ring weight from vmax for tropical cyclones"""
    if vmax < 65:
        return 0.005 # very very small
    elif vmax >= 65 and vmax < 84:
        return 0.0694
    elif vmax >= 84:
        return 0.0263
    else:
        logger.error('`op_vmax` must be a scalar')
        return 0.0

def _get_mask_val(channel_type):
    """Get mask value for a given channel type"""
    if channel_type.lower() == '89ghz':
        return 245 # 245 avoids ice
    elif channel_type.lower() in ('37ghz', '183ghz'):
        return 50 # Basically, no mask
    elif channel_type.lower() == 'ir':
        return 265 # 265: Keep low cloud only
    elif channel_type.lower() in ('swir', 'vis', 'dnb', 'rscat', 'ascat'):
        return -1e6
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return 0.0

def _get_structure_height_km(channel_type):
    """Get structure height for a given channel type"""
    if channel_type.lower() == '89ghz':
        return 10
    elif channel_type.lower() == '37ghz':
        return 3
    elif channel_type.lower() == '183ghz':
        return 3
    elif channel_type.lower() in ('ascat', 'rscat'):
        return 0
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return 0

def _get_penalty_weight(channel_type):
    """Get penalty weight for a given channel type"""
    if channel_type.lower() in ('89ghz', '37ghz', '183ghz'):
        return 1.0
    elif channel_type.lower() in ('ir', 'swir', 'vis', 'dnb'):
        return 0.33
    elif channel_type.lower() in ('ascat', 'rscat'):
        return 1.0
    else:
        logger.error(f'Unknown `channel_type`: {channel_type}')
        return 0.0

def _get_combined_score_dict(in_dict, mask_val, alpha):
    """Calculate gridded score components"""
    score_dict = score_funcs.combo_parts_calc_3_0(
        in_dict, penalty_weight=in_dict['penalty_weight'], mask_val=mask_val, alpha=alpha
    )

    # Clean up the spiral and ring score to allow a combo center in nan (including cloud-masked) areas
    np.nan_to_num(score_dict['spiral_score_grid'], nan=-1e9, copy=False)
    np.nan_to_num(score_dict['ring_score_grid'], nan=0, copy=False)

    # Calculate combo score, target point
    confidence_grid = score_dict['spiral_score_grid'] + in_dict['ring_weight'] * score_dict['ring_score_grid']
    score_dict['combo_score_grid'] = confidence_grid - score_dict['penalty_grid']
    return score_dict, confidence_grid

def _calc_confidence_score(confidence_grid, score_dict, confidence_dist_deg=0.75):
    """Calculate confidence score using combo grid w/o the distance penalty"""
    i_conf_max, j_conf_max = np.unravel_index(np.argmax(confidence_grid), confidence_grid.shape)
    lon_conf_max, lat_conf_max = score_dict['lon_grid1'][i_conf_max, j_conf_max], score_dict['lat_grid1'][i_conf_max, j_conf_max]

    # Compute distance squared without generating intermediate arrays
    lat_dist, lon_dist = score_dict['lat_grid1']-lat_conf_max, score_dict['lon_grid1']-lon_conf_max
    dist_squared_grid = lat_dist**2 + (np.cos(np.deg2rad(lat_conf_max)) * lon_dist)**2

    # The confidence score is the maximum value minus the highest score confidence_dist_deg away
    confidence_max = confidence_grid[i_conf_max, j_conf_max]
    confidence_2nd_max = np.max(confidence_grid[dist_squared_grid > confidence_dist_deg**2])
    return confidence_max - confidence_2nd_max

def _is_visir(channel_type: str) -> bool:
    """Check if the channel type is visir"""
    return channel_type.lower() in ('ir', 'swir', 'vis', 'dnb')

def archer4(
    image: dict[str, np.ndarray],
    attrib: dict[str, str | float | bool | dict[str, str | float | bool]],
    first_guess: dict[str, str | float | bool | dict[str, str | float | bool]],
    sector_info: Optional[dict[str, str | float | bool]] = None,
    alpha: float = np.deg2rad(5),
    para_fix: bool = True,
    display_filename: Optional[str] = None,
    quiet: Optional[bool] = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    This function channels the operations into one of the ARCHER variants, which have
    the same I/O requirements but different internal logic.

    Parameters
    ----------
    image: dict[str, np.ndarray]
        2D grids corresponding to the image
        image['lat_grid']:  Latitude grid. See notes.
        image['lon_grid']:  Longitude grid
        image['data_grid']: Image data (brightness temp or 0-255 brightness value)
        *image['azm_grid']: Azimuthal direction of beam (available from VIISR and ATMS Level 1B)
        *image['zen_grid']: Zenith angle of beam (available from same)
            *: Optional. Used for highest accuracy parallax calculation if desired.
    attrib: dict[str, str | float | bool | dict[str, str | float | bool]]
        Image attributes
        attrib['sat']: Satellite source (NOAA-16, Meteosat-10, F18, Aqua, GOES-16, etc). See notes.
        attrib['sensor']: Sensor instrument (SSMI, Imager, etc). See notes.
        attrib['scan_type']: 'Conical', 'Crosstrack' or 'Geo'. Used for parallax calculations
        attrib['archer_channel_type']: Name of channel according to archer rules. See notes.
        +attrib['nadir_lon']: Nadir longitude (East +).
            +: Necessary for Geo data only. Used for parallax calculation.
    first_guess: dict[str, str | float | bool | dict[str, str | float | bool]]
        First guess of the center fix
        *first_guess['source']: Source of first guess estimate (fx, bt, NHCanfx, etc). See notes.
        *first_guess['time']: datetime object
        first_guess['vmax']: Estimated Vmax of the TC.
        first_guess['lat']: Estimated first guess latitute of TC center.
        first_guess['lon']: Estimated first guess longitude of TC center.
            *: Optional. Not used yet, but probably will be brought in later.
    sector_info: dict[str, str | float | bool | dict[str, str | float | bool]]
        Dictionary available from GeoIPS (pyresample). Currently this is only used here
        for information passed on to the f-deck output string
        sector_info['storm_basin'] : Two-character basin code
        sector_info['storm_num'] : Two-digit storm code
    para_fix: bool
        Flag to control whether to apply parallax fix to the imagery. True [or False].
    display_filename: Path and filename of diagnostic display image. No image if None.

    Returns
    -------
    in_dict: dict[str, np.ndarray]
        Variables as used in ARCHER calculations, presented here for diagnostics.
        in_dict['sensor'] = Same as attrib['archer_channel_type']
        in_dict['lon_mx'] = Navigation of image used here (either parallax fixed or not)
        in_dict['lat_mx'] = "
        in_dict['bt_mx'] = Image data, either in BT or pseudo-BT
        in_dict['time'] = Time of the image
        in_dict['op_lon'] = Same as first_guess['lon']
        in_dict['op_lat'] = Same as first_guess['lat']
        in_dict['op_vmax'] = Same as first_guess['vmax']
        in_dict['ring_weight'] = Relative weight of ring score (dependent on sensor)

    out_dict: dict[str, np.ndarray]
        ARCHER output.
        out_dict['archer_channel_type'] = Same as attrib['archer_channel_type']
        out_dict['ring_radius_deg'] = Radius of resolved inner eyewall, units: deg
        out_dict['score_by_radius_arr'] = Ring score as a function of radius
        out_dict['confidence_score'] = Metric of contour spacing near center fix.
        out_dict['alpha_parameter'] = Parameter of gamma distribution (uncertainty)
        out_dict['radius50percCertDeg'] = Radius of 50% certainty area for center fix
        out_dict['radius95percCertDeg'] = Radius of 95% certainty area for center fix
        out_dict['eye_prob'] = Empirically determined probability of an eye (85GHz and IR only)
        out_dict['center_lon'] = ARCHER center fix longitude
        out_dict['center_lat'] = ARCHER center fix latitude
        out_dict['weak_center_lon'] = Last-resort center fix that violates some rules
        out_dict['weak_center_lat'] = "
        out_dict['masked'] = Whether the image was masked

    score_dict: dict[str, np.ndarray]
        Intermediate products of ARCHER, for diagnostics and dependent algorithms.
        score_dict['lon_grid1'] = Resampled regular grid coordinates
        score_dict['lat_grid1'] = Resampled regular grid coordinates
        score_dict['data_grid1'] = Resampled image
        score_dict['spiral_score_grid'] = Grid representing center of spiral shearing 
        score_dict['penalty_grid'] = Grid that exercises light penalty for straying from first guess
        score_dict['ring_score_grid'] = Grid representing center of possible eye
        score_dict['ring_radius_grid'] = Grid of candidates for ring radius
        score_dict['ring_score_grid_full'] = 3D grid (lat x lon x radius) used for ERC apps
        score_dict['radial_gradient_4d'] = 4D grid (lat x lon x 2d of gradient) for diagnostics
        score_dict['fraction_input'] = Fraction of domain covered by real image data

    Notes
    -----
    1. Image navigation rules:
        a. You must use the original navigation of the image in order to calculate parallax.
        b. Archer can calcuate parallax from a subsection (line and/or element) of a geo image
            or conical scan. However, to calculate parallax of crosstrack scans, Archer must have 
            a subsection with full lines (all the elements of the line). 
        c. Conical scans are expected to default to the format that is found in their Level 1B
            files, which is rows = lines and cols = elements. 
        d. Cross-track scans are expected to default to the opposite format, as found in *their*
            Level 1B files, which is rows = elements and cols = lines. For example, a grid of 
            ATMS data is expected to have 96 rows always, VIIRS DNB data is expected to have 
            4064 rows always, etc.
        e. The one exception to all this is if you include image['azm_grid'] and image['zen_grid'] 
            as inputs. Then archer can calculate parallax on any projection or subsection.

    2. attrib['sat']: Name of the satellite. This is not *yet* used in Archer, but may be
        necessary in the future. Follow the GeoIPS rules for satellite naming.

    3. attrib['sensor']: One of the following:
        For conical microwave sensors: ['SSMI', 'SSMIS', 'TMI', 'GMI', 'AMSRE', 'AMSR2']
        For crosstrack microwave sensors: ['AMSUB', 'MHS', 'ATMS']
        For polar imagers: ['VIIRS'] (Add AVHRR?)
        For geo imagers: ['Imager']

    4. attrib['archer_channel_type']: One of these valid strings: ['37GHz', '89GHz', '183GHz',
        'IR', 'SWIR', 'Vis', 'DNB']
        Used for directing channel-specific logic in archer.

    5. first_guess['source']: One of these strings:
        'fx': Forecast
        'bt': Best track
        'manual': Manual estimate
        'archer': Previous archer estimate
        'NHCfxan': Splice of NHC-generated analysis and forecast track
        This variable is for nothing yet, but it may be needed in the future.

    6. For more context on these methods, please refer to:
        Wimmers, A. J., and C. S. Velden, 2016: Advancements in objective multisatellite 
        tropical cyclone center fixing. J. Appl. Meteor. Climatol., 55, 197–212.
    
    History
    -------
    Developed by: Anthony J. Wimmers (2020)
    Modified by: Taiga Tsukada (2026)
    """
    in_dict = {}
    out_dict = {}
    score_dict = {}
    in_dict['ring_weight'] = _get_ring_weight(attrib['archer_channel_type'], vmax=first_guess['vmax'])
    in_dict['structure_height_km'] = _get_structure_height_km(attrib['archer_channel_type'])
    in_dict['penalty_weight'] = _get_penalty_weight(attrib['archer_channel_type'])

    if attrib['archer_channel_type'].lower() == '89ghz':
        image['bt_grid'] = map_tools.nan_around_coasts_89(image['lon_grid'], image['lat_grid'], image['data_grid'])
    elif attrib['archer_channel_type'].lower() in ('37ghz', '183ghz', 'ir', 'swir'): # Not as well validated
        image['bt_grid'] = image['data_grid']
    elif attrib['archer_channel_type'].lower() in ('vis', 'dnb'):
        if attrib['scan_type'].lower() == 'geo' and attrib['archer_channel_type'].lower() == 'vis':
            # Normalize for solar zenith angle. This matters because a difference in average brightness value from image to image does indeed throw of the results.
            cos_solar_zenith_grid = nav_tools.cos_solar_zenith(image['lon_grid'], image['lat_grid'], first_guess['time'])
            # Sun must be >1.5 degrees above horizon for that conversion to make sense
            cos_solar_zenith_grid[cos_solar_zenith_grid < 0.0262] = np.nan
            # Convert to pseudo brightness temp for Archer to act on using optimized sqrt
            bv_norm = image['data_grid'] / np.sqrt(cos_solar_zenith_grid)
            image['bt_grid'] = 350 - 0.75 * bv_norm
            # Filter out nighttime cases
            num_pix = np.product(np.shape(image['bt_grid']))
            num_nan = np.sum(np.isnan(image['bt_grid']), axis=(0,1))
            if num_nan / num_pix > 0.3:
                logger.error('Too dark for ARCHER. Exiting.')
                return in_dict, out_dict, score_dict
        else:
            # Assume polar imagery is already normalized:
            image['bt_grid'] = 350 - 0.75 * image['data_grid']
    else:
        logger.error(f'Unknown `channel_type`: {attrib["archer_channel_type"]}')
        return in_dict, out_dict, score_dict

    # Revamp lon grid to prevent being cut in half in case it straddles the antemeridian
    image['lon_grid'] = nav_tools.antemeridian_decross(image['lon_grid'], first_guess['lon'])

    # Parallax fix
    if para_fix:
        if 'zen_grid' in image.keys() and 'azm_grid' in image.keys():
            image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_allnav(image['lon_grid'], image['lat_grid'], image['zen_grid'], image['azm_grid'], in_dict["structure_height_km"])
        else:
            if attrib['scan_type'].lower() == 'geo':
                image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_geo(image['lon_grid'], image['lat_grid'], attrib['nadir_lon'], attrib['sensor'], in_dict["structure_height_km"])
            elif attrib['scan_type'].lower() == 'conical':
                image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_conical(image['lon_grid'], image['lat_grid'], attrib['sensor'], in_dict['structure_height_km'])
            elif attrib['scan_type'].lower() == 'crosstrack':
                image['lon_pc_grid'], image['lat_pc_grid'] = nav_tools.parallax_fix_crosstrack(image['lon_grid'], image['lat_grid'], attrib['sensor'], attrib['archer_channel_type'], in_dict['structure_height_km'])
            else:
                logger.error(f"Unknown scan type: {attrib['scan_type']}")
                image['lon_pc_grid'], image['lat_pc_grid'] = image['lon_grid'], image['lat_grid']
    else:
        image['lon_pc_grid'], image['lat_pc_grid'] = image['lon_grid'], image['lat_grid']

    # Reduce resolution of the image data to prevent lags and artifacts
    if attrib['archer_channel_type'].lower() in ("ir", "swir", "vis", "dnb"):
        image = nav_tools.reduce_res(image, step_km=4)

    # Write all the input data to a dictionary
    # Note here that the naming of "default" lat/lon changes between in_dict and image. 
    # image lat/lon grid is *unaltered* lat/lon. However, in_dict lat/lon is the lat/lon to be used in ARCHER.
    in_dict['sensor'] = attrib['archer_channel_type']
    in_dict['op_lon'] = first_guess['lon']
    in_dict['op_lat'] = first_guess['lat']
    in_dict['op_vmax'] = first_guess['vmax']
    in_dict['time'] = first_guess['time']
    in_dict['bt_mx'] = image['bt_grid']

    # Calculate gridded score components
    COVERAGE_THRESHOLD = 0.5

    # Unmasked with parallax correction
    mask_val = None
    in_dict['lon_mx'] = image['lon_pc_grid']
    in_dict['lat_mx'] = image['lat_pc_grid']
    score_dict_wo_mask, conf_grid_wo_mask = _get_combined_score_dict(in_dict, mask_val=mask_val, alpha=alpha)
    qc_wo_mask = score_funcs.quality_check(score_dict_wo_mask['spiral_score_grid'], score_dict_wo_mask['fraction_input'], coverage_threshold=COVERAGE_THRESHOLD)

    if qc_wo_mask and _is_visir(attrib['archer_channel_type']):
        out_dict['uses_target'] = True
        score_dict = score_dict_wo_mask
        confidence_grid = conf_grid_wo_mask
        in_dict["mask_val"] = None
        out_dict["masked"] = False
    else:
        # Try with masking but without parallax correction
        mask_val = _get_mask_val(attrib['archer_channel_type'])
        in_dict['lon_mx'] = image['lon_grid']
        in_dict['lat_mx'] = image['lat_grid']

        # With IR imagery, we have to consider pixels with partial clouds/convection.
        # Dilate these features by 2 (NOTE: for ~10 km resolution):
        if attrib['archer_channel_type'].lower() in ('ir', ):
            DILATE_VAL = 2
            in_dict['bt_mx'] = _dilate_nan(in_dict['bt_mx'], DILATE_VAL)

        score_dict_w_mask, conf_grid_w_mask = _get_combined_score_dict(in_dict, mask_val=mask_val, alpha=alpha)
        qc_w_mask = score_funcs.quality_check(score_dict_w_mask['spiral_score_grid'], score_dict_w_mask['fraction_input'], coverage_threshold=COVERAGE_THRESHOLD)

        if qc_w_mask and _is_visir(attrib['archer_channel_type']):
            out_dict['uses_target'] = True
            score_dict = score_dict_w_mask
            confidence_grid = conf_grid_w_mask
            in_dict["mask_val"] = mask_val
            out_dict["masked"] = True
        else:
            if qc_w_mask and qc_wo_mask:
                out_dict['uses_target'] = True
            else:
                out_dict['uses_target'] = False
            if np.max(score_dict_w_mask["combo_score_grid"]) > np.max(score_dict_wo_mask["combo_score_grid"]):
                score_dict = score_dict_w_mask
                confidence_grid = conf_grid_w_mask
                in_dict["mask_val"] = mask_val
                out_dict["masked"] = True
            else:
                in_dict['lon_mx'] = image['lon_pc_grid']
                in_dict['lat_mx'] = image['lat_pc_grid']
                score_dict = score_dict_wo_mask
                confidence_grid = conf_grid_wo_mask
                in_dict["mask_val"] = None
                out_dict["masked"] = False

    # Determine the location of the maximum combo score
    i_max_score, j_max_score = np.unravel_index(np.argmax(score_dict['combo_score_grid']), score_dict['combo_score_grid'].shape)
    lon_max_score, lat_max_score = score_dict['lon_grid1'][i_max_score, j_max_score], score_dict['lat_grid1'][i_max_score, j_max_score]

    # Extract ring radius and score at this location
    out_dict['ring_radius_deg'] = score_dict['ring_radius_grid'][i_max_score, j_max_score]
    ring_score_max = score_dict['ring_score_grid'][i_max_score, j_max_score]
    out_dict['score_by_radius_arr'] = np.squeeze(score_dict['ring_score_grid_full'][i_max_score, j_max_score, :])
    # gradient_grid = np.squeeze(score_dict['radial_gradient_4d'][i_max_score, j_max_score, :])

    # Represent the center fix uncertainty in terms of radii of 50% and 95% confidence
    out_dict['confidence_score'] = _calc_confidence_score(confidence_grid, score_dict, confidence_dist_deg=0.75)
    out_dict['alpha_parameter'] = conversions.confidence_to_alpha(out_dict['confidence_score'], attrib['archer_channel_type'], 0, in_dict['op_vmax'])
    radius_err_range = np.arange(0, 10, 0.01) # in degrees
    cdf = 1.0 - (out_dict['alpha_parameter'] * radius_err_range + 1.0) * np.exp(-out_dict['alpha_parameter'] * radius_err_range)

    RECORDED_CONFIDENCE_LEVELS = [0.50, 0.95]
    for level in RECORDED_CONFIDENCE_LEVELS:
        nearest_idx = np.argmin(np.abs(cdf - level))
        radius_err_at_level = radius_err_range[nearest_idx]
        out_dict['radius' + str(int(level * 100)) + 'percCertDeg'] = radius_err_at_level

    # Calculate the probability of having detected an eye (only for 89GHz)
    if attrib['archer_channel_type'].lower() in ('89ghz', 'ir'):
        if attrib['archer_channel_type'].lower() == '89ghz':
            calib_stat_arr = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70,  75,  80]
            calib_perc_arr = [0, 9, 27, 44, 56, 72, 82, 90, 94, 99, 100, 100]
        elif attrib['archer_channel_type'].lower() == 'ir':
            calib_stat_arr = [0, .1, .5,  1, 1.5,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  13,  14,  15]
            calib_perc_arr = [0,  1,  7, 15,  40, 47, 50, 60, 65, 70, 75, 80, 85, 95, 98, 99, 100, 100, 100]
        eye_prob_stat = out_dict['confidence_score'] * ring_score_max
        out_dict['eye_prob'] = np.interp(eye_prob_stat, calib_stat_arr, calib_perc_arr)
    else:
       out_dict['eye_prob'] = None

    # Pack up the output values
    out_dict['archer_channel_type'] = attrib['archer_channel_type']
    
    # Store the center fix in the output dictionary
    if out_dict['uses_target']: # This is an official center-fix
        out_dict['center_lon'] = nav_tools.antemeridian_restore(lon_max_score)
        out_dict['center_lat'] = lat_max_score
        out_dict['weak_center_lon'] = None
        out_dict['weak_center_lat'] = None
    else: # This is a center-fix if you must, but it's not official because it's probably corrupted
        out_dict['center_lon'] = None
        out_dict['center_lat'] = None
        out_dict['weak_center_lon'] = nav_tools.antemeridian_restore(lon_max_score)
        out_dict['weak_center_lat'] = lat_max_score

    # Plot diagnostic figure if display_filename is not None
    if display_filename is not None:
        import matplotlib.pyplot as plt
        fig = plot_tools.plot_diag_4panel(image, attrib, in_dict, out_dict, score_dict, display_filename=display_filename)
        plt.close(fig)

    # Produce an fdeck-formatted string
    fdeck_str = fdeck_tools.generate_string(attrib, in_dict, out_dict, sector_info=sector_info)
    out_dict['fdeck_string'] = fdeck_str
    if not quiet:
        logger.info(fdeck_str)

    return in_dict, out_dict, score_dict