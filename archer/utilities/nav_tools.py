#%%
import os
import logging

import numpy as np
import pandas as pd
from pyproj import Geod


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def antemeridian_decross(lon_grid, first_guess_lon):
    """
    Normalize longitude grid to consistent hemisphere based on first guess location.
    
    If TC is near antemeridian (±170°), wrap all longitudes to the same side:
    - first_guess_lon < 0 (Western hemisphere): use -180 to 0
    - first_guess_lon >= 0 (Eastern hemisphere): use 0 to 180
    """
    crosses_antemeridian = np.any(lon_grid > 170) and np.any(lon_grid < -170)
    
    if not crosses_antemeridian:
        return lon_grid
        
    # Western hemisphere
    if first_guess_lon < 0:
        return np.where(lon_grid > 100, lon_grid - 360, lon_grid)
    # Eastern hemisphere
    else:
        return np.where(lon_grid < -100, lon_grid + 360, lon_grid)

def antemeridian_restore(lon):
    """Insure that the longitude value is between -180 and 180"""
    lon = (lon + 180) % 360 - 180
    return lon

def reduce_step(lon_grid, lat_grid, step_km=4):
    """
    Calculate the row and col step necessary to bring the image to step_km resolution
    For IR, step_km is 4 (km)
    """
    km_per_gcd = np.deg2rad(6370)
    num_rows, num_cols = np.shape(lon_grid)

    mid_row = int(round(num_rows / 2))
    mid_col = int(round(num_cols / 2))

    # Use the middle of the grid as the reference
    row_dist_km = km_per_gcd * np.hypot( 
        ((lon_grid[mid_row, mid_col] - lon_grid[mid_row+1, mid_col]) / np.cos(lat_grid[mid_row, mid_col])),
        (lat_grid[mid_row, mid_col] - lat_grid[mid_row+1, mid_col]))
    col_dist_km = km_per_gcd * np.hypot( 
        ((lon_grid[mid_row, mid_col] - lon_grid[mid_row, mid_col+1]) / np.cos(lat_grid[mid_row, mid_col])),
        (lat_grid[mid_row, mid_col] - lat_grid[mid_row, mid_col+1]))

    # Use "floor" to keep it conservative
    new_row_step = int(max(1, np.floor(step_km / row_dist_km)))
    new_col_step = int(max(1, np.floor(step_km / col_dist_km)))
    return new_row_step, new_col_step

def reduce_res(image, step_km=4):
    """
    Reduce the resolution of the image by factors of new_row_step and new_col_step
    """
    new_row_step, new_col_step = reduce_step(image['lon_grid'], image['lat_grid'], step_km=step_km)

    if new_row_step > 1 or new_col_step > 1:
        for key in image.keys():
            image[key] = image[key][::new_row_step, ::new_col_step]
    return image

def extrap_row1(xMx, yMx):
    xNm2 = xMx[:,2]
    xNm1 = xMx[:,1]
    xN = xMx[:,0]
    yNm2 = yMx[:,2]
    yNm1 = yMx[:,1]
    yN = yMx[:,0]

    xNp = 2 * xNm1 - xNm2
    yNp = 2 * yNm1 - yNm2

    xNp1p = 2 * xN - xNm1
    yNp1p = 2 * yN - yNm1

    alpha = np.arctan2(yN-yNm1, xN-xNm1) - np.arctan2(yNp-yNm1, xNp-xNm1)

    xNp1 = xN + (xNp1p-xN)*np.cos(alpha) - (yNp1p-yN)*np.sin(alpha)
    yNp1 = yN + (xNp1p-xN)*np.sin(alpha) + (yNp1p-yN)*np.cos(alpha)
    return xNp1, yNp1

def extrap_rowN(xMx, yMx):
    xNm2 = xMx[:,-3]
    xNm1 = xMx[:,-2]
    xN = xMx[:,-1]
    yNm2 = yMx[:,-3]
    yNm1 = yMx[:,-2]
    yN = yMx[:,-1]

    xNp = 2 * xNm1 - xNm2
    yNp = 2 * yNm1 - yNm2

    xNp1p = 2 * xN - xNm1
    yNp1p = 2 * yN - yNm1

    alpha = np.arctan2(yN - yNm1, xN - xNm1) - np.arctan2(yNp - yNm1, xNp - xNm1)

    xNp1 = xN + (xNp1p - xN) * np.cos(alpha) - (yNp1p - yN) * np.sin(alpha)
    yNp1 = yN + (xNp1p - xN) * np.sin(alpha) + (yNp1p - yN) * np.cos(alpha)
    return xNp1, yNp1

def parallax_fix_conical(lon_grid, lat_grid, sensor, structure_height_km):
    km_per_gcd = np.deg2rad(6370)
    if sensor.lower() == 'ssmi':
        view_angle_deg = 53.1
        flip_elements = False
    elif sensor.lower() == 'ssmis':
        view_angle_deg = 53.1
        flip_elements = True
    elif sensor.lower() == 'tmi':
        view_angle_deg = 53.1
        flip_elements = False
    elif sensor.lower() in ('amsre', 'amsr2'):
        view_angle_deg = 55.2
        flip_elements = True
    elif sensor.lower() == 'gmi':
        view_angle_deg = 52.8
        flip_elements = True
    else:
        logger.error(f'Unknown sensor: {sensor}')
        return lon_grid, lat_grid

    # Calculate nudges and nudge the image to make a first-approximation correction for
    # parallax *according to the most important features in the image*
    if flip_elements:
        lon_grid = lon_grid[:,::-1]
        lat_grid = lat_grid[:,::-1]

    lonCol0, latCol0 = extrap_row1(lon_grid, lat_grid)
    # Use hstack and newaxis instead of the deprecated np.matrix approach array concat
    lon_grid = np.hstack([lonCol0[:, np.newaxis], lon_grid])
    lat_grid = np.hstack([latCol0[:, np.newaxis], lat_grid])

    lonColN1, latColN1 = extrap_rowN(lon_grid, lat_grid)
    lon_grid = np.hstack([lon_grid, lonColN1[:, np.newaxis]])
    lat_grid = np.hstack([lat_grid, latColN1[:, np.newaxis]])

    # Calculate parallax offset based on scanline orientations and view angle
    dist2p = structure_height_km * np.tan(np.deg2rad(view_angle_deg)) / km_per_gcd
    cos13 = np.cos(np.deg2rad(0.5 * (lat_grid[:, :-2] + lat_grid[:, 2:])))
    delX13 = cos13 * (lon_grid[:, :-2] - lon_grid[:, 2:])
    delY13 = lat_grid[:, :-2] - lat_grid[:, 2:]
    dist13 = np.hypot(delX13, delY13)

    # Apply offset
    lon_grid = lon_grid[:, 1:-1] - dist2p * delY13 / dist13 / cos13
    lat_grid = lat_grid[:, 1:-1] + dist2p * delX13 / dist13

    if flip_elements:
        lon_grid = lon_grid[:,::-1]
        lat_grid = lat_grid[:,::-1]
    return lon_grid, lat_grid


def parallax_fix_geo(lon_grid, lat_grid, nadir_lon, sensor, structure_height_km):
    r_earth = 6370
    r_geo = 35788
    xre = r_earth * np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
    yre = r_earth * np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
    zre = r_earth * np.sin(np.deg2rad(lat_grid))
    xrd = (r_geo+r_earth) * np.cos(np.deg2rad(nadir_lon)) - xre
    yrd = (r_geo+r_earth) * np.sin(np.deg2rad(nadir_lon)) - yre
    zrd = -zre
    remag = np.sqrt((xre**2) + (yre**2) + (zre**2))
    rdmag = np.sqrt((xrd**2) + (yrd**2) + (zrd**2))
    cosz = (xre*xrd + yre*yrd + zre*zrd) / (remag*rdmag)
    delX_m = 1000 * structure_height_km * np.tan(np.arccos(cosz))

    # Calculate the direction of nudging to correct for parallax
    geod = Geod(ellps='sphere')
    _, azimuth21, _ = geod.inv(
        nadir_lon * np.ones(np.shape(lon_grid)), 0 * np.ones(np.shape(lon_grid)), 
        lon_grid, lat_grid)
    
    # Apply the nudge
    new_lon_grid, new_lat_grid, _ = geod.fwd(lon_grid, lat_grid, azimuth21, delX_m)
    return new_lon_grid, new_lat_grid

def parallax_fix_crosstrack(
    lon_grid, lat_grid, sensor, archer_channel_type, structure_height_km):

    this_dir = os.path.dirname(os.path.realpath(__file__))
    etc_dir = os.path.join(this_dir, '../etc/')

    if sensor.lower() == 'atms':
        scan_angle_arr = np.abs(np.linspace(-52.77, 52.77, 96))
        nadir_lon_arr = np.mean(lon_grid[:, 47:48], axis=1) # Correct row/col?
        nadir_lat_arr = np.mean(lat_grid[:, 47:48], axis=1)
        scan_angle_grid, nadir_lon_grid = np.meshgrid(scan_angle_arr, nadir_lon_arr)
        _              , nadir_lat_grid = np.meshgrid(scan_angle_arr, nadir_lat_arr)
    elif sensor.lower() in ('amsub', 'mhs'):
        scan_angle_arr = pd.read_csv(
            os.path.join(etc_dir, 'amsub90scanangles.csv'), header=None)
        nadir_lon_arr = np.mean(lon_grid[:, 44:45], axis=1)
        nadir_lat_arr = np.mean(lat_grid[:, 44:45], axis=1)
        scan_angle_grid, nadir_lon_grid = np.meshgrid(scan_angle_arr, nadir_lon_arr)
        _              , nadir_lat_grid = np.meshgrid(scan_angle_arr, nadir_lat_arr)
    elif sensor.lower() == 'viirs' and archer_channel_type.lower() == 'dnb':
        scan_angle_arr = pd.read_csv(
            os.path.join(etc_dir, 'viisr_4064scanangles.csv'), header=None)
        nadir_lon_arr = np.mean(lon_grid[:, 2031:2032], axis=1)
        nadir_lat_arr = np.mean(lat_grid[:, 2031:2032], axis=1)
        scan_angle_grid, nadir_lon_grid = np.meshgrid(scan_angle_arr, nadir_lon_arr)
        _              , nadir_lat_grid = np.meshgrid(scan_angle_arr, nadir_lat_arr)
    elif sensor.lower() == 'viirs':
        scan_angle_arr = pd.read_csv(
            os.path.join(etc_dir, 'viisr_6400scanangles.csv'), header=None)
        nadir_lon_arr = np.mean(lon_grid[:, 3199:3200], axis=1)
        nadir_lat_arr = np.mean(lat_grid[:, 3199:3200], axis=1)
        scan_angle_grid, nadir_lon_grid = np.meshgrid(scan_angle_arr, nadir_lon_arr)
        _              , nadir_lat_grid = np.meshgrid(scan_angle_arr, nadir_lat_arr)
    else:
        logger.error(f'Unknown sensor: {sensor}')
        return lon_grid, lat_grid

    # Calculate the distance of nudging to correct for parallax
    delX_m = 1000 * structure_height_km * np.tan(np.deg2rad(scan_angle_grid))

    # Calculate the direction of nudging to correct for parallax
    geod = Geod(ellps='sphere')
    _, azimuth21, _ = geod.inv(nadir_lon_grid, nadir_lat_grid, lon_grid, lat_grid)

    # Apply the nudge
    new_lon_grid, new_lat_grid, _ = geod.fwd(lon_grid, lat_grid, azimuth21, delX_m)
    return new_lon_grid, new_lat_grid

def parallax_fix_allnav(lon_grid, lat_grid, zenGrid, azmGrid, structure_height_km):
    # Calculate the distance of nudging to correct for parallax
    delX_m = 1000 * structure_height_km * np.tan(np.deg2rad(zenGrid))

    # Apply the nudge in the opposite of the azimuthal direction
    geod = Geod(ellps='sphere')
    new_lon_grid, new_lat_grid, _ = geod.fwd(lon_grid, lat_grid, azmGrid, -delX_m)
    return new_lon_grid, new_lat_grid

def cos_solar_zenith(lon_grid, lat_grid, datetime):
    """
    Calculates cosine of solar zenith angle over a grid of points
    Reference: http://en.wikipedia.org/wiki/Insolation
    Note: The declination calculation on the wiki page looks like it's in
    error. It uses the angle of orbit, which looks out of place in the
    calculations. I used my own calculation for "del", and it checks out on
    the equinox/solstice days.
    AJW (2011)
    """
    # Relevant times
    img_time = pd.to_datetime(datetime)
    hr_float = img_time.hour + img_time.minute/60.0 + img_time.second/3600.0

    # Terms from the wiki Insolation page:
    # Obliquity of the earth (degrees)
    eta = 23.4398

    # Declination of the earth (degrees)
    delta = eta * np.sin(np.deg2rad((284 + img_time.dayofyear) / 365.0 * 360.0))

    # Angle relative to peak insolation longitude (hours*deg/hr)
    h = lon_grid + (hr_float - 12.0) * 15.0 
    
    # Pre-calculate trigonometric arrays to avoid repetitive deg2rad over full grids
    lat_rad = np.deg2rad(lat_grid)
    delta_rad = np.deg2rad(delta)
    
    cos_solar_zenith_grid = np.sin(lat_rad) * np.sin(delta_rad) + \
        np.cos(lat_rad) * np.cos(delta_rad) * np.cos(np.deg2rad(h))
    return cos_solar_zenith_grid