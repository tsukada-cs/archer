#%%
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s p%(process)d] %(message)s', datefmt='%d%b %H%M:%S')


def pcolorCenterShift(xGrid, yGrid, zGrid):
    xGrid = np.pad(xGrid, ((1,1), (1,1)), mode='edge')
    yGrid = np.pad(yGrid, ((1,1), (1,1)), mode='edge')
    zGrid = np.pad(zGrid, ((1,1), (1,1)), mode='edge')
    return xGrid, yGrid, zGrid

def pcolorCenterShiftDataOnly(zGrid):
    zGrid = np.pad(zGrid, ((1,1), (1,1)), mode='edge')
    return zGrid

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Credit to: Jake VanderPlas
    # https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    # Usage example: plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
    base = cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))

    # Force the first element to be gray (for satellite source plots):
    color_list[0] = (8./15., 8./15., 8./15., 1.0)
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def get_channel_settings(image, attrib):
    # Channel-specific settings
    if attrib['archer_channel_type'] == '89GHz':
        disp_grid0 = image['bt_grid']
        color_lo = [160, 160, 160]
        color_hi = [280, 280, 280]
        color_label = 'Brightness temperature (K)'
    elif attrib['archer_channel_type'] == '37GHz':
        disp_grid0 = image['bt_grid']
        color_lo = [160, 160, 160]
        color_hi = [280, 280, 280]
        color_label = 'Brightness temperature (K)'
    elif attrib['archer_channel_type'] == '183GHz':
        disp_grid0 = image['bt_grid']
        color_lo = [160, 160, 160]
        color_hi = [280, 280, 280]
        color_label = 'Brightness temperature (K)'
    elif attrib['archer_channel_type'] == 'IR':
        disp_grid0 = image['bt_grid']
        color_lo = [180, 180, 180]
        color_hi = [300, 300, 300]
        color_label = 'Brightness temperature (K)'
    elif attrib['archer_channel_type'] == 'SWIR':
        disp_grid0 = image['bt_grid']
        color_lo = [180, 180, 180]
        color_hi = [300, 300, 300]
        color_label = 'Brightness temperature (K)'
    elif attrib['archer_channel_type'] == 'Vis' or attrib['archer_channel_type'] == 'DNB':
        disp_grid0 = image['data_grid']
        color_lo = [0,   150, 150]
        color_hi = [255, 320, 320]
        color_label = 'Pseudo brightness temperature (K)'
    else:
        logger.error('No real channel type to display')
    return disp_grid0, color_lo, color_hi, color_label

def plot_diag_4panel(
    image: Dict[str, np.ndarray], 
    attrib: Dict[str, str], 
    in_dict: Dict[str, np.ndarray], 
    out_dict: Dict[str, np.ndarray], 
    score_dict: Dict[str, np.ndarray], 
    display_filename: Optional[str] = None, 
    cmap: str = 'bone'
) -> None:
    # Prepare grids for plotting
    disp_grid0, color_lo, color_hi, color_label = get_channel_settings(image, attrib)
    ring_weight = 1 # in_dict['ring_weight'] # Helps you see something

    lon_mx_disp, lat_mx_disp, bt_mx_disp = pcolorCenterShift(in_dict['lon_mx'], in_dict['lat_mx'], disp_grid0)
    lon_grid_disp, lat_grid_disp, bt_grid_disp = pcolorCenterShift(score_dict['lon_grid1'], score_dict['lat_grid1'], score_dict['data_grid1'])

    i_ss_max, j_ss_max = np.unravel_index(np.nanargmax(score_dict['spiral_score_grid']), score_dict['spiral_score_grid'].shape)
    ss_max_lon = score_dict['lon_grid1'][i_ss_max, j_ss_max]
    ss_max_lat = score_dict['lat_grid1'][i_ss_max, j_ss_max]

    i_rs_max, j_rs_max = np.unravel_index(np.nanargmax(score_dict['ring_score_grid']), score_dict['ring_score_grid'].shape)
    rs_max_lon = score_dict['lon_grid1'][i_rs_max, j_rs_max]
    rs_max_lat = score_dict['lat_grid1'][i_rs_max, j_rs_max]

    fig, ax = plt.subplots(2, 2, figsize=(7.5, 8), gridspec_kw=dict(hspace=0.0))
    op_scatter_kwargs = dict(s=80, marker="+", c="#333333", lw=1.0, zorder=2.1)
    spiral_scatter_kwargs = dict(s=40, marker="o", ec="green", fc='none', lw=0.8, zorder=2.1)
    ring_scatter_kwargs = dict(s=40, marker="x", c="purple", lw=0.8, zorder=2.1)
    final_scatter_kwargs = dict(s=40, marker="s", ec="k", fc='none', lw=1.2, zorder=2.1)

    # Storm synoptic view
    ax.flat[0].pcolormesh(lon_grid_disp, lat_grid_disp, bt_grid_disp, vmin=color_lo[0], vmax=color_hi[0], cmap=cmap)
    ax.flat[0].scatter(in_dict['op_lon'], in_dict['op_lat'], **op_scatter_kwargs)
    ax.flat[0].scatter(ss_max_lon, ss_max_lat, **spiral_scatter_kwargs)
    ax.flat[0].scatter(rs_max_lon, rs_max_lat, **ring_scatter_kwargs)
    ax.flat[0].scatter(out_dict['center_lon'], out_dict['center_lat'], **final_scatter_kwargs)
    ax.flat[0].set(xlim=(np.min(lon_grid_disp[0,:]), np.max(lon_grid_disp[0,:])), ylim=(np.min(lat_grid_disp[:,0]), np.max(lat_grid_disp[:,0])))
    ax.flat[0].set_title('(a) Center fix synoptics', loc='left')

    # Storm combined score view
    ax.flat[1].pcolormesh(lon_grid_disp, lat_grid_disp, bt_grid_disp, vmin=color_lo[2], vmax=color_hi[2], cmap=cmap)

    if out_dict['uses_target']:
        combo_score_grid_disp = score_dict['combo_score_grid']
        with np.errstate(invalid='ignore'):
            combo_score_grid_disp[combo_score_grid_disp < -1e8] = np.nan
        ax.flat[1].contour(
            score_dict['lon_grid1'], score_dict['lat_grid1'], combo_score_grid_disp, 
            levels=10, linewidths=0.6, cmap="Reds"
        )
        ax.flat[1].scatter(out_dict['center_lon'], out_dict['center_lat'], **final_scatter_kwargs)
        ax.flat[1].scatter(in_dict['op_lon'], in_dict['op_lat'], **op_scatter_kwargs)
    ax.flat[1].set(xlim=(np.min(lon_grid_disp[0,:]), np.max(lon_grid_disp[0,:])), ylim=(np.min(lat_grid_disp[:,0]), np.max(lat_grid_disp[:,0])))
    ax.flat[1].set_title('(b) Combined score', loc='left')    
    

    # Storm spiral view
    ax.flat[2].pcolormesh(lon_mx_disp, lat_mx_disp, bt_mx_disp, vmin=color_lo[0], vmax=color_hi[0], cmap=cmap)
    #img1 = ax.flat[1].scatter(lon_mx, lat_mx, c=bt_noco_mx, vmin=160, vmax=280, cmap=cmap, edgecolors='none', marker=',')
    spiral_score_grid_disp = score_dict['spiral_score_grid']
    with np.errstate(invalid='ignore'):
        spiral_score_grid_disp[spiral_score_grid_disp < -1e8] = np.nan
    ax.flat[2].contour(
        score_dict['lon_grid1'], score_dict['lat_grid1'], spiral_score_grid_disp, 
        levels=10, linewidths=0.6, cmap="Reds"
    )
    ax.flat[2].scatter(ss_max_lon, ss_max_lat, **spiral_scatter_kwargs)
    ax.flat[2].scatter(in_dict['op_lon'], in_dict['op_lat'], **op_scatter_kwargs)

    aspect_lon = np.cos(np.deg2rad(in_dict['op_lat'])) # Aspect ratio
    ax.flat[2].set(xlim=(ss_max_lon-2.5/aspect_lon, ss_max_lon+2.5/aspect_lon), ylim=(ss_max_lat-2.5, ss_max_lat+2.5))
    ax.flat[2].set_title(f'(c) Spiral score', loc='left')

    rect = mpatches.Rectangle(
        (np.min(lon_grid_disp), np.min(lat_grid_disp)), 
        np.max(lon_grid_disp) - np.min(lon_grid_disp), 
        np.max(lat_grid_disp) - np.min(lat_grid_disp), 
        linewidth=1, edgecolor='k', facecolor='none', ls="--", label="Zoomed region"
    )
    ax.flat[2].add_patch(rect)

    # Storm ring score view
    ax.flat[3].pcolormesh(lon_grid_disp, lat_grid_disp, bt_grid_disp, vmin=color_lo[1], vmax=color_hi[1], cmap=cmap)

    ring_score_grid_disp = score_dict['ring_score_grid']
    with np.errstate(invalid='ignore'):
        ring_score_grid_disp[ring_score_grid_disp == 0] = np.nan
    ax.flat[3].contour(
        score_dict['lon_grid1'], score_dict['lat_grid1'], ring_weight * ring_score_grid_disp, 
        levels=5, linewidths=0.6, cmap="Reds"
    )
    ax.flat[3].set(xlim=(np.min(lon_grid_disp[0,:]), np.max(lon_grid_disp[0,:])), ylim=(np.min(lat_grid_disp[:,0]), np.max(lat_grid_disp[:,0])))
    ax.flat[3].set_title('(d) Ring score', loc='left')
    ax.flat[3].scatter(rs_max_lon, rs_max_lat, **ring_scatter_kwargs)
    ax.flat[3].scatter(in_dict['op_lon'], in_dict['op_lat'], **op_scatter_kwargs)

    if out_dict['uses_target']:
        circle = mpatches.Circle(
            (out_dict['center_lon'], out_dict['center_lat']),
            out_dict['ring_radius_deg'],
            edgecolor='m',
            facecolor='none',
            lw=1.5,
            zorder=2.1,
        )
        ax.flat[3].add_patch(circle)

    # Colorbar
    p = ax.flat[2].get_position()
    cax = fig.add_axes([p.x0, p.y0-0.019, p.width, 0.015])
    cax.tick_params(direction="in")
    cbar = fig.colorbar(ax.flat[2].collections[0], cax=cax, orientation='horizontal')
    cbar.set_label(color_label)

    # Legend
    ax.flat[3].scatter([], [], marker='s', edgecolor='k', facecolor='none', ls=(0,(3,2)), lw=0.8, label="Zoomed region")
    ax.flat[3].scatter([], [], **op_scatter_kwargs, label='Operational fix')
    ax.flat[3].scatter([], [], **final_scatter_kwargs, label='ARCHER fix')
    ax.flat[3].scatter([], [], **spiral_scatter_kwargs, label='Spiral max')
    ax.flat[3].scatter([], [], **ring_scatter_kwargs, label='Ring max')
    ax.flat[3].scatter([], [], marker='o', edgecolor='m', facecolor='none', lw=1.5, label="Ring")
    ax.flat[3].legend(
        frameon=False,
        loc='upper center',
        ncol=2,
        bbox_to_anchor=(0.45, -0.05),
        columnspacing=1.5,
        handletextpad=0.3,
    )

    # Suptitle
    time_str = datetime.fromtimestamp(in_dict["time"]).strftime('%Y-%m-%d %H:%M UTC')
    title = f'[{attrib["archer_channel_type"]}] {time_str}  Vmax = {in_dict["op_vmax"]} kt'
    fig.suptitle(title, y=0.91)

    # All axes
    for i, iax in enumerate(ax.flat):
        iax.set(aspect='equal')
        iax.tick_params(direction="in", right=True, top=True)
        
    if display_filename is not None:
        fig.savefig(display_filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
        logger.info(f'Saved: {os.path.abspath(display_filename)}')
    else:
        plt.show()
    return fig