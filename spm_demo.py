#%%
import os

import numpy as np
import xarray as xr

from archer.archer4_TT import archer4

fpath = "/home/tsukada/git/archer/data/spm/WP022023_202305250420_UNHF3o4v5.nc"
# fpath = "/home/tsukada/git/archer/data/spm/WP232024_202410300500_UNHF3o4v5.nc"
# fpath = "/home/tsukada/git/archer/data/spm/WP232024_202410280500_UNHF3o4v5.nc"
# fpath = "/home/tsukada/git/archer/data/spm/WP022023_202305220400_UNHF3o4v5.nc"
spm = xr.open_dataset(fpath)
lat2d, lon2d = np.meshgrid(spm['latitude'], spm['longitude'], indexing='ij')

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass
# %%
%%time
image = {}
image['data_grid'] = spm['tb_89h'].values
image['lat_grid'] = lat2d
image['lon_grid'] = lon2d

attrib = {}
attrib['sat'] = 'Aqua'
attrib['sensor'] = 'SPM'
attrib['scan_type'] = 'Conical'
attrib['archer_channel_type'] = '89GHz'

first_guess = {}
first_guess['source'] = 'best'
first_guess['time'] = spm.time.values
first_guess['vmax'] = float(spm.attrs['tcIntensity'].replace("knots",""))
first_guess['lat'] = float(spm.attrs['tcLat'][:-1])
first_guess['lon'] = float(spm.attrs['tcLon'][:-1])

sector_info = {}
sector_info["storm_basin"] = spm.attrs["tcBasin"]
sector_info["storm_num"] = spm.attrs["tcNumber"]

img_opath = f"imgs/{os.path.basename(fpath).replace('.nc', '.png')}"
in_dict, out_dict, score_dict = archer4(
    image, 
    attrib, 
    first_guess, 
    sector_info=sector_info,
    alpha=np.deg2rad(5), 
    para_fix=False,
    display_filename=img_opath
)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.pcolormesh(lon2d, lat2d, spm['tb_89h'], shading='auto', cmap='bone')
ax.scatter(out_dict['archer_lon'], out_dict['archer_lat'], 50, ec="k", fc="none", marker='s')
ax.scatter(in_dict['op_lon'], in_dict['op_lat'], 80, "k", marker='+')
ax.set(xlim=(score_dict['lon_grid1'].min(), score_dict['lon_grid1'].max()),
       ylim=(score_dict['lat_grid1'].min(), score_dict['lat_grid1'].max()))
ax.set(aspect="equal")
# %%
