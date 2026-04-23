# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:56:54 2016

@author: wimmers
"""
import logging

import numpy as np
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')

def interp_swath_to_global_rect_gridset(lonSwath, latSwath, dataSwath, footprintSwath, timeSwath, lonGridArr, latGridArr):
    """
    Special case of nearest-neighbor interpolation for three of the same grids
    """
    # Make the "index swath"
    numScans, numElems = np.shape(latSwath)
    indexSwath = np.arange(0, numScans*numElems).reshape(numScans, numElems)
    
    # Interpolate to the "index grid"
    indexInterpGrid = interp_swath_to_global_rect_grid(lonSwath, latSwath, indexSwath, lonGridArr, latGridArr, 'nearest')

    # These are the grid indeces to work from
    IsInIndexGrid = np.logical_not( np.isnan(indexInterpGrid) )
    
    # These are the swath indeces to pull into the output grids
    indexSwathList = indexInterpGrid[IsInIndexGrid].astype(int)
 
    # Create the output grids in each the same way
    dataInterpGrid = np.nan * np.zeros( np.shape(indexInterpGrid) )
    dataInterpGrid[IsInIndexGrid] = dataSwath.flatten()[indexSwathList]

    footprintInterpGrid = np.nan * np.zeros( np.shape(indexInterpGrid) )
    footprintInterpGrid[IsInIndexGrid] = footprintSwath.flatten()[indexSwathList]

    timeInterpGrid = np.nan * np.zeros( np.shape(indexInterpGrid) )
    timeInterpGrid[IsInIndexGrid] = timeSwath.flatten()[indexSwathList]
    return dataInterpGrid, footprintInterpGrid, timeInterpGrid

def interp_swath_to_global_rect_grid(lonSwath, latSwath, dataSwath, lonGridArr, latGridArr, interpType):
    """
    Computes an interpoated rectangular grid from a single swath, parsing the swath by zone (north pole,
    midlatitude, south pole) in order to get a fast and accurate interpolation in all zones.  
    
    Inputs:
        lonSwath, latSwath: Navigation of the swath
        dataSwath: Data, such as TPW
        lonGridArr, latGridArr: 1D arrays of the new grid navigation. NOTE! lonGridArr
            must not have redundant lons on the left and right. Use -180 and 179.75, for example.
        interpType: Interpolation type for griddata - 'linear', 'cubic', 'nearest', etc.
            NOTE! Currently 'cubic' does not work b/c it conflicts with the nan mask. Use
            'linear' or 'nearest' instead.
    
    Returns:
        dataInterpGrid: Interpolated grid of data in the grid navigation
    Feb 2016 (AJW)
    """
    # Useful constants
    numScans, numElems = np.shape(latSwath)
    centerElem = int(numElems / 2)
    
    # Initialize output array
    dataInterpGrid = np.full((np.size(latGridArr), np.size(lonGridArr)), np.nan)
    
    # Divide the swath into overlapping zone sections
    zoneCodeArr = np.zeros((numScans, 1)) # Midlatitude (rectangular)
    zoneCodeArr[latSwath[:,centerElem] >  60] =  1 # North (polar)
    zoneCodeArr[latSwath[:,centerElem] < -60] = -1 # South (polar)

    # Iterate by scan, and operate on a new section when we get to the end of it.
    # Allow the first and last scans to just be part of their neighboring scans.
    sectionInitIdx = 0
    for scanIdx in range(1, numScans-1):
        # This finds the end of the ongoing section (North polar, midlatitude or south polar)
        if zoneCodeArr[scanIdx] != zoneCodeArr[scanIdx-1] or scanIdx == numScans-3:
            sectionEndIdx = scanIdx + 2
            zoneCode = zoneCodeArr[scanIdx-1]
            #print 'Start Idx: ', sectionInitIdx, ', Final Idx: ', sectionEndIdx
            #print 'Zone Code: ', zoneCode
            
            # Interpolate this section in the proper way to the rectangular grid
            dataInterpSection = interp_section_to_global_rect_grid(
                lonSwath[sectionInitIdx:sectionEndIdx, :],
                latSwath[sectionInitIdx:sectionEndIdx, :],
                dataSwath[sectionInitIdx:sectionEndIdx, :],
                lonGridArr, latGridArr, interpType, zoneCode)           
            
            # Add the new interpolated section to the grid
            dataInterpGrid[np.logical_not(np.isnan(dataInterpSection))] = \
                dataInterpSection[np.logical_not(np.isnan(dataInterpSection))]
            # Set the new start index for the next section
            sectionInitIdx = scanIdx
    return dataInterpGrid

def interp_section_to_global_rect_grid(lonSection, latSection, dataSection, lonGridArr, latGridArr, interp_type='cubic', zoneCode=0):
    """
    Computes an interpoated rectangular grid from a single swath section, according to its zone (north pole,
    midlatitude, south pole).
    
    Inputs:
        lonSection, latSection: Navigation of the swath section
        dataSection: Data, such as TPW
        lonGridArr, latGridArr: 1D arrays of the new grid navigation
        interpType: Interpolation type for griddata - 'linear', 'cubic', 'nearest', etc.
        zoneCode: 1 (North pole zone), 0 (Midlatiude zone), or -1 (South pole zone)
    
    Returns:
        dataInterpSection: Interpolated grid of section's data in the grid navigation
        
    Feb 2016 (AJW)
    """
    # Useful constants
    numScans, numElems = np.shape(latSection)
    centerElem = int(numElems / 2)
    centerScan = int(numScans / 2)
    lonGridRes = np.abs(lonGridArr[1] - lonGridArr[0])
    BUFFER_XY = 1.5 # Interpolate around the borders of the swath by this much 
    
    # Initialize grid and section variables
    lonGrid, latGrid = np.meshgrid(lonGridArr, latGridArr)
    dataInterpSection = np.full(np.shape(lonGrid), np.nan)
    
    # Convert according to zone:
    if zoneCode == 0:
        # Convert to a swath centered around zero
        centerLon = np.round(lonSection[centerScan, centerElem]/lonGridRes) * lonGridRes
        xSection = (lonSection - centerLon + 180) % 360 - 180
        ySection = latSection
        
        # Work out the reordering for grids as well
        xGridOrigArr = (lonGridArr - centerLon + 180) % 360 - 180
        centeredLonOrder = np.argsort(xGridOrigArr)
        centeredLonReorder = np.argsort(centeredLonOrder)

        # Keep *grid* in original coordinates, but then rearrange with 
        # centeredLonReorder at the end
        xGrid = lonGrid
        yGrid = latGrid
    elif zoneCode == 1:
        # Convert from polar to rectangular coordinates
        xSection = (90 - latSection) * np.cos(np.deg2rad(lonSection))
        ySection = (90 - latSection) * np.sin(np.deg2rad(lonSection))
        xGrid = (90 - latGrid) * np.cos(np.deg2rad(lonGrid))
        yGrid = (90 - latGrid) * np.sin(np.deg2rad(lonGrid))
    elif zoneCode == -1:
        # Convert from polar to rectangular coordinates
        xSection = (90 + latSection) * np.cos(np.deg2rad(lonSection))
        ySection = (90 + latSection) * np.sin(np.deg2rad(lonSection))
        xGrid = (90 + latGrid) * np.cos(np.deg2rad(lonGrid))
        yGrid = (90 + latGrid) * np.sin(np.deg2rad(lonGrid))
    else:
        raise ValueError(f'Unknown zone code: {zoneCode}')
    
    # Do the interpolation with cKDTree on the subsection defined by inSubGrid
    x_flat = xSection.ravel()
    y_flat = ySection.ravel()
    d_flat = dataSection.ravel()
    
    valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat) | np.isnan(d_flat))
    
    inSubGrid = np.logical_and( 
        np.logical_and(xGrid > np.nanmin(xSection) - BUFFER_XY, 
                       xGrid < np.nanmax(xSection) + BUFFER_XY), 
        np.logical_and(yGrid > np.nanmin(ySection) - BUFFER_XY, 
                       yGrid < np.nanmax(ySection) + BUFFER_XY))
                       
    try:
        if np.any(valid_mask):
            tree = cKDTree(np.column_stack((x_flat[valid_mask], y_flat[valid_mask])))
            query_pts = np.column_stack((xGrid[inSubGrid].ravel(), yGrid[inSubGrid].ravel()))
            
            # Use distance upper bound to prevent artifacts (equivalent to the old nan envelope)
            max_dist = 1.5 * lonGridRes
            
            if interp_type == 'nearest':
                dists, idxs = tree.query(query_pts, k=1, distance_upper_bound=max_dist)
                valid_queries = dists < np.inf
                
                if np.any(valid_queries):
                    res = np.full(query_pts.shape[0], np.nan)
                    res[valid_queries] = d_flat[valid_mask][idxs[valid_queries]]
                    dataInterpSection[inSubGrid] = res
            else:
                # 'linear' or 'cubic' -> IDW
                dists, idxs = tree.query(query_pts, k=3, distance_upper_bound=max_dist)
                valid_queries = dists[:, 0] < np.inf
                
                if np.any(valid_queries):
                    res = np.full(query_pts.shape[0], np.nan)
                    v_dists = dists[valid_queries]
                    v_idxs = idxs[valid_queries]
                    
                    eps = 1e-12
                    weights = 1.0 / (v_dists**2 + eps)
                    
                    valid_neighbors = v_dists < np.inf
                    weights[~valid_neighbors] = 0.0
                    
                    safe_idxs = np.where(valid_neighbors, v_idxs, 0)
                    vals = d_flat[valid_mask][safe_idxs]
                    
                    res[valid_queries] = np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)
                    dataInterpSection[inSubGrid] = res
        else:
            dataInterpSection[inSubGrid] = np.nan
    except Exception as e:
        logger.error(f'Error: Could not interpolate with cKDTree on this section: {e}')
        dataInterpSection[inSubGrid] = np.nan
    
    # Shift back from recentering if remapped in zoneCode 0 (midlatitudes)
    if zoneCode == 0:
        dataInterpSection = dataInterpSection[:, centeredLonReorder]
    return dataInterpSection