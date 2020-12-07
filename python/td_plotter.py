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
td_plots = False
line_plots = True

driver_time = ['P275']
mag_str = ['B50']
Amp = ['A40']

#driver_time = ['P50','P200','P300']
#mag_str = ['B50']
#Amp = ['A20','A40','A60','A80']
#tilt = ['T5','T10','T15','T45','T60']

sim_nbs = len(driver_time)*len(mag_str)*len(Amp)

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/'
file_names = np.empty(sim_nbs, dtype='U30')
#save_folder = path_2_shared_drive+'python/td_plot_figs/'
#if save_figs==True:
#    Path(save_folder).mkdir(parents=True, exist_ok=True)

cmap_array = ['YlOrRd','hot', 'bwr', 'bwr']

counter = 0
for nb_dt in range(len(driver_time)):
    for nb_mag in range(len(mag_str)):
        for nb_amp in range(len(Amp)):
            file_names[counter] = 'jet_'+driver_time[nb_dt]+'_'+mag_str[nb_mag]+'_'+Amp[nb_amp]+'_'
            counter += 1
axis_title_size = 20
tick_size = 18
f_size = (20,26)

for fname in file_names:
    name_parts =  fname.split('_')[:-1]
    join_name = '_'.join([fi for fi in name_parts]) 
    save_folder = path_2_shared_drive+'python/td_plot_figs_sharc/'+join_name+'/'
    if save_figs==True:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
    path_td_data = 'td_plots_data_sharc/'+fname
    h_range = glob.glob(path_td_data[:-1]+'/*')
    for HoI in range(len(h_range)):
        csv_name = 'data.csv'    
        path_2_files = path_td_data[:-1]+'/'+str(HoI+1)+'Mm/'+csv_name 
        
        df = pd.read_csv(path_2_files) 
        heading_names = list(df)
        
        one_data_len = np.argwhere(np.diff(df['time [s]'])>0,)[0][0]+1 
        grid_dims = [int(df['time [s]'].size/one_data_len), one_data_len]
        
        time_2d_grid = np.reshape(df[heading_names[0]].values, grid_dims)
        x_2d_grid = np.reshape(df[heading_names[1]].values, grid_dims)
        rho_2d_grid = np.reshape(df[heading_names[2]].values, grid_dims)
        Te_2d_grid = np.reshape(df[heading_names[3]].values, grid_dims)
        Vx_2d_grid = np.reshape(df[heading_names[4]].values, grid_dims)
        Vy_2d_grid = np.reshape(df[heading_names[5]].values, grid_dims)
        data_cluster = [[rho_2d_grid, Te_2d_grid],[Vx_2d_grid, Vy_2d_grid]]
        
        if td_plots == True:        
            f, ax = plt.subplots(2, 2, figsize=f_size)
            f.set_size_inches(32, 18)
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())

        if line_plots==True:
            line_fig, lin_ax = plt.subplots(figsize=f_size)
            line_fig.set_size_inches(32, 18)
        
        c = 0
        if td_plots == True:
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
                
        if line_plots==True:
            lin_ax.set_xlabel('time (s)')
            lin_ax.set_ylabel('Density^2 [kg m-3]^2', color='red')
            lin_ax.plot(time_2d_grid[1:,0], np.mean(rho_2d_grid[1:]**2-rho_2d_grid[:-1]**2,axis=1), '-r')
#            lin_ax.tick_params(axis='y', labelcolor='red')
                        
            lin_ax2 = lin_ax.twinx()
            lin_ax2.set_ylabel('Width [km]', color='blue')
            lin_ax2.plot(df['time [s]'][1:], df['width [km]'][1:].values-df['width [km]'][:-1].values, '--b')
            
            if save_figs==True:
                line_fig.savefig(save_folder+fname+'_line_plot_'+str(HoI+1)+'Mm.png')
                line_fig.clf()
                plt.close()

