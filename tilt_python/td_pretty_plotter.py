import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#import matplotlib.image as mpimg
import os
import numpy as np
#from PIL import Image
#import img2vid as i2v
import glob
#import yt
#from yt.units import second
from pathlib import Path
import pandas as pd
from itertools import chain 

def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return at


def side_pts_of_jet(clipped_data, jet_height, nb_pts, DOMIAN, shape):
    # gets multiple pts on jet side at one instance of time
    # clipped_data - data set to scan 
    # jet_height - index value of the jet height
    # DOMIAN - phyiscal length of whole domain (before clipping)
    # shape - shape of whole domain before clipping
    indexs_of_slices = np.linspace(0, jet_height[1], nb_pts, dtype=int)
    js_idx_x = []
    js_val_x = []
    js_idx_y = []
    js_val_y = []
    for idx in indexs_of_slices:
        dumvar1,dumvar2,dumvar3,dumvar4 = side_pts(clipped_data, idx, DOMIAN, shape)
        # restructiong data
        # all x values
        js_idx_x.append(dumvar1[0])
        js_idx_x.append(dumvar2[0])
        js_val_x.append(dumvar3[0])
        js_val_x.append(dumvar4[0])
        # all y data
        js_idx_y.append(dumvar1[1])
        js_idx_y.append(dumvar2[1])
        js_val_y.append(dumvar3[1])
        js_val_y.append(dumvar4[1])
    return js_idx_x, js_idx_y, js_val_x, js_idx_y

degree_sign= u'\N{DEGREE SIGN}'
save_figs = True
quad_image = False
mono_image = True

#SMALL_SIZE = 48
#MEDIUM_SIZE = 50
#BIGGER_SIZE = 52
SMALL_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titl


#driver_time = ['P300']
#mag_str = ['B50']
#Amp = ['A80']
#tilt = ['T30']

csv_name = 'data.csv'

driver_time = ['P300']
mag_str = ['B60']
Amp = ['A60']
tilt = ['T0','T5','T10','T15','T20','T25', 'T30', 'T35', 'T40', 'T45', 'T50', 'T55', 'T60']

#driver_time = ['P50','P200','P300']
#mag_str = ['B50']
#Amp = ['A20','A40','A60','A80']
#tilt = ['T5','T10','T15','T45','T60']

cbar_den_lims = [2e-7, 1.5e-7, 1.5e-7, 1.0e-7]
y_lmb_ul = [550, 500, 500, 450]
sim_nbs = len(tilt)*len(driver_time)*len(mag_str)*len(Amp)
xlim = np.array((-1, 1))
nrows = 4
ncols = 3

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/'
file_names = np.empty(sim_nbs, dtype='U80')
#save_folder = path_2_shared_drive+'tilt_python/sharc_run/sj_td_plot_figs/'
#if save_figs==True:
#    Path(save_folder).mkdir(parents=True, exist_ok=True)
cmap_array = ['YlOrRd','hot', 'bwr', 'bwr']

counter = 0
for nb_dt in range(len(driver_time)):
    for nb_tilt in range(len(tilt)):
        for nb_mag in range(len(mag_str)):
            for nb_amp in range(len(Amp)):
                file_names[counter] = 'jet_'+driver_time[nb_dt]+'_'+mag_str[nb_mag]+'_'+Amp[nb_amp]+'_'+tilt[nb_tilt]
                counter += 1
axis_title_size = 20
tick_size = 18
f_size = (20,26)
image_range = np.arange(0, 12+1, 1)
shared_jet_name = 'jet_P300_B60_A60_T*/'
data_fname = '/*_data.csv'
heights = [1] # Mm
for HoI in heights:
    path_td_data = 'sharc_run/td_plots_data_sj/'+shared_jet_name+str(HoI)+'Mm/'+csv_name
    file_oI = np.asarray(glob.glob(path_td_data))
    tilt_list = []
    for file in file_oI:
        tilt_list.append(int(file.split('/')[-3].split('T')[-1]))
    file_oI = file_oI[np.argsort(tilt_list)] 
#    col_time_2d_grid = []
#    col_x_2d_grid = []
    time_min_list = []
    time_max_list = []
    x_min_list = []
    x_max_list = []
    shape_list = []
    for pic_idx in image_range:
        df = pd.read_csv(file_oI[pic_idx])
        heading_names = list(df)
        time_min_list.append(min(df[heading_names[0]].values))
        time_max_list.append(max(df[heading_names[0]].values))
        x_min_list.append(min(df[heading_names[1]].values))
        x_max_list.append(max(df[heading_names[1]].values))
    time_extent = max(time_min_list), min(time_max_list)
    x_extent = max(x_min_list), min(x_max_list)
    aspect_ratio = sum(abs(xlim))/sum(abs(np.asarray(time_extent)))
    extent = x_extent[0], x_extent[-1], time_extent[0], time_extent[-1]

    col_rho_2d_grid = []
    for pic_idx in image_range:
        df = pd.read_csv(file_oI[pic_idx])
        idx_select = np.argwhere((df['time [s]'].values<=time_extent[-1]) & (df['time [s]'].values>=time_extent[0]))
        df = df.loc[0:idx_select[-1][0]]
        heading_names = list(df)
        one_data_len = np.argwhere(np.diff(df['time [s]'])>0,)[0][0]+1 
        grid_dims = [int(df['time [s]'].size/one_data_len), one_data_len]
        rho_2d_grid = np.reshape(df[heading_names[2]].values, grid_dims)
        shape_list.append(np.shape(rho_2d_grid))
        col_rho_2d_grid.append(rho_2d_grid)

    F = plt.figure(figsize=(32, 18))
    F.clf()
    grid2 = ImageGrid(F, 111,
                      nrows_ncols=(nrows, ncols),
                      direction="row",
                      axes_pad=0.0,
                      add_all=True,
                      label_mode="1",
                      share_all=True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="3%",
                      cbar_pad=0.05,
                      )
    lab_loc = nrows*ncols-ncols
    grid2[lab_loc].set_xlabel("X [Mm]")
    grid2[lab_loc].set_ylabel("Height [Mm]")

    vmin_list = []
    vmax_list = []
    for nb_array in col_rho_2d_grid:
        vmin_list.append(np.min(nb_array))
        vmax_list.append(np.max(nb_array))
    vmax, vmin = np.max(vmax_list), np.min(vmin_list)    
#    vmax, vmin = cbar_den_lims[1], np.min(vmin_list)    
    import matplotlib.colors
    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)

    for ax, z in zip(grid2, col_rho_2d_grid):
        im = ax.imshow(z, norm=norm, cmap=cmap_array[0],
                       origin="lower", extent=extent,
                       interpolation="nearest", aspect=aspect_ratio)
        ax.set_xlim(-1,1)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    
    grid2[0].cax.colorbar(im)
    cax = grid2.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text("Density $[kg ~m^{-3}]$")
    
    for ax, im_title in zip(grid2, [r'$\theta=$0'+degree_sign,
                                    r'$\theta=$5'+degree_sign,
                                    r'$\theta=$10'+degree_sign,
                                    r'$\theta=$15'+degree_sign,
                                    r'$\theta=$20'+degree_sign,
                                    r'$\theta=$25'+degree_sign,
                                    r'$\theta=$30'+degree_sign,
                                    r'$\theta=$35'+degree_sign,
                                    r'$\theta=$40'+degree_sign,
                                    r'$\theta=$45'+degree_sign,
                                    r'$\theta=$50'+degree_sign,
                                    r'$\theta=$55'+degree_sign]):
        t = add_inner_title(ax, im_title, loc='upper left')
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    F.savefig('test_td_plot_1Mm.png', bbox_inches='tight')

    plt.draw()
    plt.show()
