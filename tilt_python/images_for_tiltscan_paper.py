import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mplt_col
import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import pandas as pd 
from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")
import amrvac_pytools as apt


def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
#    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
#                size=plt.rcParams['legend.fontsize'])
    prop = dict(size=plt.rcParams['legend.fontsize']-8)
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return at

THETA = r'$\theta$'
degree_sign= u'\N{DEGREE SIGN}'
SMALL_SIZE = 36
MEDIUM_SIZE = SMALL_SIZE + 4
BIGGER_SIZE = SMALL_SIZE + 2 
#plt.rc('axes', edgecolor='white')
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-3)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure titl

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'
T_degs = ['30','35','40','45','50','55']

path_2_files = ['/j/T/P300/B60/A60/T'+T_degs[0]+'/', '/j/T/P300/B60/A60/T'+T_degs[1]+'/', '/j/T/P300/B60/A60/T'+T_degs[2]+'/',
                '/j/T/P300/B60/A60/T'+T_degs[3]+'/', '/j/T/P300/B60/A60/T'+T_degs[4]+'/', '/j/T/P300/B60/A60/T'+T_degs[5]+'/']
file_names = ['jet_P300_B60_A60_T'+T_degs[0]+'_', 'jet_P300_B60_A60_T'+T_degs[1]+'_', 'jet_P300_B60_A60_T'+T_degs[2]+'_',
              'jet_P300_B60_A60_T'+T_degs[3]+'_', 'jet_P300_B60_A60_T'+T_degs[4]+'_', 'jet_P300_B60_A60_T'+T_degs[5]+'_']

# constants
unit_length = 1e9  # cm
DOMIAN = [5*unit_length, 3*unit_length]
unit_temperature = 1e6  # K
unit_numberdensity = 1e9  # cm^-3

g_cm3_to_kg_m3 = 1e3
dyne_cm2_to_Pa = 1e-1
cm_to_km = 1e-5
cm_to_Mm = 1e-8
s_to_min = 1/60

unit_density = 2.3416704877999998E-015
unit_velocity = 11645084.295622544
unit_pressure = 0.31754922400000002
unit_magenticfield = 1.9976088799077159
unit_time = unit_length/unit_velocity

# lowdt
dt = unit_time/20
unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

save_fig =True
dat_file = False
vtu_file = True
B_plots = True
test_image = False
check_dex = 0
if B_plots == True:
# mag tension is not okay at lower bc
    shift_x_factor = 2.4e2
else:
    shift_x_factor = 0

nrows = 2
ncols = 3

xres = 4096
yres = 2944
grid_size = np.divide(DOMIAN,[yres, xres])
physical_grid_size_xy =  DOMIAN/np.array([xres,yres])*cm_to_Mm

#t0 = 0
#t1 = 140
#nb_steps = 141

t0 = 114
t1 = 140
nb_steps = 26

var_names = ['density']
cmaps = ['gist_heat']

time_stamps = np.linspace(t0, t1, nb_steps, dtype=int)
F = plt.figure(figsize=(30, 24))
physical_time = np.round(time_stamps*dt, 1)
FIRST = True
for ind, ti in enumerate(time_stamps):
    clipped_rho_data = []
    offset = str(ti).zfill(4)
    for stuff in range(len(file_names)):
        file_name = file_names[stuff]
        path_2_file = path_2_files[stuff]
        Full_path = path_2_shared_drive + path_2_file + file_name
        if dat_file==True:
            ds = apt.load_datfile(Full_path+offset+'.dat')
            ad = ds.load_all_data()            
            var_rho_data = ad['rho']*unit_density
        if vtu_file==True:
            ds = apt.load_vtkfile(ti, file=Full_path, type='vtu')
    
            var_rho_data = apt.vtkfiles.rgplot(ds.rho, data=ds, cmap='hot')
            var_rho_data, x_grid0, y_grid0 = var_rho_data.get_data(xres=xres, yres=yres)
            var_rho_data = var_rho_data*g_cm3_to_kg_m3
            plt.close('all')
            shape = np.shape(var_rho_data)
            if FIRST == True:
                #These don't work for first time step
                indexs_x = np.nonzero(x_grid0[:,0])[0]
                mid_pt_x = int(round((min(indexs_x)+(max(indexs_x)-min(indexs_x))/2)))
    #            mid_pt_x = 2067 
                clip_range_x = round(0.05*shape[0]) 
                scan_range_x = [mid_pt_x-clip_range_x, mid_pt_x+clip_range_x]
                scan_range_y = [0, round(shape[1]/2)]

                clipped_grid_x = np.rot90(x_grid0[scan_range_x[0]:scan_range_x[-1],
                                          scan_range_y[0]:scan_range_y[-1]])
                clipped_grid_y = np.rot90(y_grid0[scan_range_x[0]:scan_range_x[-1],
                                          scan_range_y[0]:scan_range_y[-1]])
                x_extent = [np.min(clipped_grid_x)*cm_to_Mm, np.max(clipped_grid_x)*cm_to_Mm]
                y_extent = [np.min(clipped_grid_y)*cm_to_Mm, np.max(clipped_grid_y)*cm_to_Mm]
                extent = x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]
                FIRST = False
        clipped_rho_data.append(np.rot90(var_rho_data[scan_range_x[0]:scan_range_x[-1],
                                  scan_range_y[0]:scan_range_y[-1]]))    
    if test_image == True:
        plt.imshow(clipped_rho_data[check_dex], extent = [x_extent[0], x_extent[1], y_extent[0],y_extent[1]])
        plt.xlim(-1.25,1.25)
        plt.ylim(0,8)
        plt.gca().set_aspect(0.75, adjustable='box')
        plt.xlabel('x (Mm)')#, fontweight="bold")
        plt.ylabel('y (Mm)')#, fontweight="bold")
        plt.show()
    grid2 = ImageGrid(F, 111,
              nrows_ncols=(nrows, ncols),
              direction="row",
              axes_pad=0.075,
              add_all=True,
              label_mode="1",
              share_all=True,
              cbar_location="right",
              cbar_mode="single",
              cbar_size="10%",
              cbar_pad=0.05)
    lab_loc = nrows*ncols-ncols
    grid2[lab_loc].set_xlabel("X [Mm]")
    grid2[lab_loc].set_ylabel("Y [Mm]")
    vmax, vmin = 1.5e-10*g_cm3_to_kg_m3, 4.5e-15*g_cm3_to_kg_m3 
    norm = mplt_col.Normalize(vmax=vmax, vmin=vmin)
    dex_num = 0
    for ax, z in zip(grid2, clipped_rho_data):
        im = ax.imshow(z, norm=norm, cmap=cmaps[0],
                       origin="upper", extent=extent,
                       interpolation="nearest", aspect=0.75)
        ax.set_xlim(-1.25,1.25)
        ax.set_ylim(0,8)
        if dex_num == 0:
              ax.set_title('t = '+str(physical_time[ind])+' s',
                           color='k', fontsize=SMALL_SIZE)
              dex_num = 1
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    
    grid2[0].cax.colorbar(im)
    cax = grid2.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text("Density $[kg ~m^{-3}]$")
    plt.rcParams.update({'text.color': "white"})
    plt.rcParams.update({"font.weight": "normal"})
    plt.rcParams.update({"axes.labelweight": "normal"})
    for ax, im_title in zip(grid2, [THETA + '= ' + T_degs[0] + degree_sign,
                                    THETA + '= ' + T_degs[1]  + degree_sign,
                                    THETA + '= ' + T_degs[2]  + degree_sign,
                                    THETA + '= ' + T_degs[3]  + degree_sign,
                                    THETA + '= ' + T_degs[4]  + degree_sign,
                                    THETA + '= ' + T_degs[5]  + degree_sign]):
        t = add_inner_title(ax, im_title, loc='upper left')
        t.patch.set_ec("none")
        t.patch.set_alpha(1)
    Path('sharc_run/fig_for_paper/tilt_scan/T_'+T_degs[0]+'_'+T_degs[1]+'_'+T_degs[2]+'_'+T_degs[3]+'_'+T_degs[4]+'_'+T_degs[5]).mkdir(parents=True, exist_ok=True)
    F.savefig('sharc_run/fig_for_paper/tilt_scan/T_'+T_degs[0]+'_'+T_degs[1]+'_'+T_degs[2]+'_'+T_degs[3]+'_'+T_degs[4]+'_'+T_degs[5]+'/tilt_scan_for_movie'+offset+'.png', bbox_inches='tight')
    F.clf()
plt.close()
