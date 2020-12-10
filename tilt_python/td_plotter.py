import sys
import matplotlib.pyplot as plt
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


save_figs = True
quad_image = False
mono_image = True

#SMALL_SIZE = 48
#MEDIUM_SIZE = 50
#BIGGER_SIZE = 52
SMALL_SIZE = 70
MEDIUM_SIZE = 72
BIGGER_SIZE = 76
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titl


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
for fname in file_names:
#    name_parts =  fname.split('_')[:-1]
#    join_name = '_'.join([fi for fi in name_parts])

    path_td_data = 'sharc_run/td_plots_data_sj/'+fname
#    h_range = glob.glob(path_td_data+'/*')
    h_range = glob.glob(path_td_data+'/2Mm')
    for HoI in range(len(h_range)):
        save_folder = path_2_shared_drive + 'tilt_python/sharc_run/sj_td_plot_sfigs/' + fname + '/' + str(HoI+1) + 'Mm'
        if save_figs == True:
            Path(save_folder).mkdir(parents=True, exist_ok=True)    
        path_2_files = path_td_data+'/'+str(HoI+1)+'Mm/'+csv_name 
        
        df = pd.read_csv(path_2_files) 
        heading_names = list(df)
        
        one_data_len = np.argwhere(np.diff(df['time [s]'])>0,)[0][0]+1 
        grid_dims = [int(df['time [s]'].size/one_data_len), one_data_len]
        
        time_2d_grid = np.reshape(df[heading_names[0]].values, grid_dims)
        x_2d_grid = np.reshape(df[heading_names[1]].values, grid_dims)
        rho_2d_grid = np.reshape(df[heading_names[2]].values, grid_dims)
        if mono_image:
            f, ax = plt.subplots(figsize=f_size)
#            f.set_size_inches(32, 18)
            f.set_size_inches(32, 18)
            im = ax.pcolormesh(x_2d_grid, time_2d_grid,
                                     rho_2d_grid, cmap=cmap_array[0])#, vmin=0, vmax=cbar_den_lims[HoI])
            cb = f.colorbar(im, ax=ax)
            cb.set_label(label='Density [kg m-3]')
            ax.set_ylabel('Time [s]')
            ax.set_xlabel(heading_names[1])
#            ax.set_ylim(ymin=np.min(time_2d_grid),ymax=y_lmb_ul[HoI])
            ax.set_xlim(-1,1)
#            plt.tight_layout()
            if save_figs==True:
                f.savefig(save_folder+'/'+fname+'_'+str(HoI+1)+'Mm.png')#,bbox_inches='tight')
                f.clf()
                plt.close()

        if quad_image == True:        
            Te_2d_grid = np.reshape(df[heading_names[3]].values, grid_dims)
            Vx_2d_grid = np.reshape(df[heading_names[4]].values, grid_dims)
            Vy_2d_grid = np.reshape(df[heading_names[5]].values, grid_dims)
            
            data_cluster = [[rho_2d_grid, Te_2d_grid],[Vx_2d_grid, Vy_2d_grid]]
            

            f, ax = plt.subplots(2, 2, figsize=f_size)
            f.set_size_inches(32, 18)
            #mng = plt.get_current_fig_manager()
            #mng.resize(*mng.window.maxsize())
                
            c = 0
            for j in range(len(data_cluster[0])):
                for k in range(len(data_cluster[1])):
                    f.suptitle(fname, size=26)
                    if j==1:
                        lim = np.max(abs(data_cluster[j][k]))
                        im = ax[j][k].pcolormesh(x_2d_grid, time_2d_grid, data_cluster[j][k], cmap=cmap_array[c], vmin=-lim, vmax=lim )
                    else:
                        im = ax[j][k].pcolormesh(x_2d_grid, time_2d_grid, data_cluster[j][k], cmap=cmap_array[c])
                    cb = f.colorbar(im, ax=ax[j][k])
                    cb.set_label(label=heading_names[c+2], size=axis_title_size)
                    cb.ax.tick_params(labelsize=tick_size)
                    ax[j][k].set_ylabel(heading_names[0], size=axis_title_size)
                    ax[j][k].set_xlabel(heading_names[1], size=axis_title_size)
                    ax[j][k].tick_params(axis='both', which='minor', labelsize=tick_size)
                    ax[j][k].tick_params(axis='both', which='major', labelsize=tick_size)
                    ax[j][k].set_xlim(-1,1)
                    c += 1
            if save_figs==True:
                f.savefig(save_folder+fname+'_'+str(HoI+1)+'Mm.png')#,dpi=600)
                f.clf()
                plt.close()

