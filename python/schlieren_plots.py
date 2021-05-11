import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg') # revert above
import matplotlib.pyplot as plt
import os
import numpy as np
#from PIL import Image
#import img2vid as i2v
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import yt
from yt.units import second
import pickle
import pandas as pd
from findiff import FinDiff

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

#from amrvac_pytools.datfiles.reading import amrvac_reader
#from amrvac_pytools.vtkfiles import read, amrplot

import amrvac_pytools as apt

def gradinet_calc(z_grid,x_grid,y_grid,n):
    # cals abs for x and y compents of delrho
    dx = np.transpose(x_grid)[0][1]-np.transpose(x_grid)[0][0] 
    dy = y_grid[0][1]-y_grid[0][0]

    dn_dxn = FinDiff(0, dx, n)     
    dn_dyn = FinDiff(1, dy, n)
    
    dnz_dxn = dn_dxn(z_grid)
    dnz_dyn = dn_dyn(z_grid)    
    return [dnz_dxn,dnz_dyn]

def sch_plot_abs_grad(rho,dx,dy):
    # cals abs for x and y compents of delrho
    grad_rho = np.gradient(rho,dx,dy)
    return np.sqrt((grad_rho[0])**2+(grad_rho[1])**2)

def convert_2_schlieren(rho, x_grid, y_grid, npgrad=True, sch=None):
    # calcs sch
    c0 = -0.001
    c1 = 0.05
    ck = 5
    if npgrad==True:
        abs_del_rho_x_y = sch_plot_abs_grad(rho,np.transpose(x_grid)[0],y_grid[0])
    else:
        grads_x_y = gradinet_calc(rho,x_grid,y_grid,1)
        abs_del_rho_x_y = np.sqrt((grads_x_y[0])**2+(grads_x_y[1])**2)
    if sch == None:
        S = (abs_del_rho_x_y-c0*abs_del_rho_x_y.max())/(c1*abs_del_rho_x_y.max()-c0*abs_del_rho_x_y.max())
    else:
        S = (abs_del_rho_x_y-c0*sch)/(c1*sch-c0*sch)  
    return np.exp(-ck*S)


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
#dir_paths =  glob.glob('../2D/P300/B50/A60')
dir_paths =  glob.glob('../B/P300/B60/A60')
#dir_paths =  glob.glob('../B/P*/B*/A*')
#dir_paths =  glob.glob('../hight_dt/P*/B*/A*')
#dir_paths = [dir_paths[29]]

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

dt = unit_time/20

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

# otpions
data_save = False
schlieren_values = True
dynamic_schlieren_plot = False
schlieren_plot = True
#extra_name = 'zoomed'
extra_name = ''
path_2_data = '\data\max_delrho_dt.dat'
yt_plot = True
Te_grad_plot = False
B_plots =True

if B_plots == True:
# mag tension is not okay at lower bc
    shift_x_factor = 2.4e2
else:
    shift_x_factor = 0

xres = 4096
yres = 2944

if yt_plot == True:
    arr_sch = np.zeros((xres, yres, 1))
    bbox = np.array([[-2.5e4-shift_x_factor, 2.5e4-shift_x_factor], [0, 3e4], [-1, 1]])
#    y_limit = 2e4
    shift_factor = 0 # km
#    shift_factor = 4000 # km

if schlieren_plot==True:
    pd_data = pd.read_pickle(path_2_data)
    schlieren_value = pd_data['max gradrho'].max()


peak_hi = 0

big_data = []
data = []
big_data_indexs = []
for path in dir_paths:
    FIRST = True
    path_parts = path.split('/')
    path_parts = path_parts[len(path_parts)-3:]
    path_numerics = np.zeros(len(path_parts))

    for j, item in enumerate(path_parts):
        path_numerics[j] = float(item[1:])

    root_save_dir = 'jet_'+path_parts[0]+'_'+path_parts[1]+'_'+path_parts[2]+'/'
    full_paths = glob.glob(path+'/jet_'+path_parts[0]+'_'+path_parts[1]+'*.vtu')
    # skip first step as no value
#    full_paths = full_paths[1:]
#    full_paths = [full_paths[1]]
#    full_paths = [full_paths[51]]
    
   
    sub_data_1 = []
    grad_max = []
    physical_time = []

    for ind, path_element in enumerate(full_paths):
        Full_path = path_2_shared_drive + path_element[2:-8]
        # need to fix ti
        ti = int(path_element[-8:-4])           
        # Reading vtu file, allows to set custum grid poitns
        ds0 = apt.load_vtkfile(ti, file=Full_path, type='vtu')
        data0 = apt.vtkfiles.rgplot(ds0.rho, data=ds0, cmap='hot')
        plt.close()
        rho_data, x_grid0, y_grid0 = data0.get_data(xres=xres, yres=yres)
#        # testing of grads
#        trho = rho_data[int(xres/2):int(xres/2)+4,0:4]
#        tx = x_grid0[int(xres/2):int(xres/2)+4,0:4]
#        ty = y_grid0[int(xres/2):int(xres/2)+4,0:4]
#
#        dt_dx_eq = np.diff(trho,axis=0)
#        dt_dy_eq = np.diff(trho,axis=1)
#
#        np_grad_trho = np.gradient(trho)
#        np_grad_trho_with_d = np.gradient(trho,np.transpose(tx)[0],ty[0])
#
#        t_diff = gradinet_calc(trho,tx,ty,1)
        if dynamic_schlieren_plot==True:
            cmap = 'gray'
            dyn_sch = convert_2_schlieren(rho_data,sch=None)
            plt.imshow(dyn_sch,cmap=cmap)
            plt.show()
        if schlieren_plot==True:
            cmap = 'gray'
            sch = convert_2_schlieren(rho_data, x_grid=x_grid0, 
                                           y_grid=y_grid0, sch=schlieren_value)
            sch_diff = convert_2_schlieren(rho_data,x_grid=x_grid0, y_grid=y_grid0,
                                      npgrad=False, sch=schlieren_value)
#            plt.imshow(sch,cmap=cmap)
#            plt.show()
#            # useless
#            laplacian_rho_1 = np.gradient(rho_data,np.transpose(x_grid0)[0],
#                                            y_grid0[0])
#            laplacian_rho_d2d2x = np.gradient(laplacian_rho_1[0],
#                                              np.transpose(x_grid0)[0],
#                                              y_grid0[0])[0]
#            laplacian_rho_d2d2y = np.gradient(laplacian_rho_1[1],
#                                              np.transpose(x_grid0)[0],
#                                              y_grid0[0])[1]
#            laplacian_rho = np.sqrt((laplacian_rho_d2d2x)**2+(laplacian_rho_d2d2y)**2)
#            lap_rho_diff = gradinet_calc(rho_data,x_grid=x_grid0, y_grid=y_grid0,n=2)
#            tot_rho_test = np.sqrt((lap_rho_diff[0])**2+(lap_rho_diff[1])**2)
            if yt_plot == True:
                arr_sch[:, :, 0] = sch
                data = dict(numerical_schlieren=(arr_sch))
                ds = yt.load_uniform_grid(data, arr_sch.shape, length_unit="km",
                     bbox=bbox, nprocs=128)
                ds.periodicity = (True, True, True)
                y_limit =  0.8e4 # km
#                y_limit =  0.2e4 # km
                slc = yt.SlicePlot(ds, "z", ['numerical_schlieren'], center=[0.0, y_limit/2+shift_factor, 0],
#                       width=((1e8, 'cm'), (y_limit*1e5, 'cm')),
                       width=((2.5e8, 'cm'), (y_limit*1e5, 'cm')),
                       origin=(0, 0, 'domain'), fontsize=52)
                slc.set_cmap('numerical_schlieren', 'gray')
                slc.set_log('numerical_schlieren', False)
                slc.set_axes_unit("Mm")
                save_folder = root_save_dir+'t'+str(ti).zfill(4)+'/'
                Path(save_folder).mkdir(parents=True, exist_ok=True)
                ds.current_time = yt.units.yt_array.YTQuantity(ti*dt*second)
#                slc.annotate_timestamp(corner='upper_left', redshift=False, draw_inset_box=True)
#                slc.annotate_timestamp(redshift=False, draw_inset_box=True,coord_system='figure')
                slc.annotate_title('Time: '+str(np.round(ds.current_time.value, 2))+' '+str(ds.current_time.units))            
                slc.save(save_folder+'T_'+str(ti).zfill(4)+'_'+extra_name)
                if Te_grad_plot == True:
                    dst = apt.load_vtkfile(ti, file=Full_path, type='vtu')
                    data0 = apt.vtkfiles.rgplot(dst.T, data=dst, cmap='hot')
                    plt.close()
                    Te_data, x_grid0, y_grid0 = data0.get_data(xres=xres, yres=yres)
                    Te_grad = sch_plot_abs_grad(Te_data,np.transpose(x_grid0)[0],
                                            y_grid0[0])
                    arr_sch[:, :, 0] = Te_grad
                    data = dict(gradient_magnitude_Te=(arr_sch))
                    ds = yt.load_uniform_grid(data, arr_sch.shape, length_unit="km",
                                              bbox=bbox, nprocs=128)
                    ds.periodicity = (True, True, True)
#                    y_limit = 2e4
                    slc = yt.SlicePlot(ds, "z", ['gradient_magnitude_Te'],
                                       center=[0.0, y_limit/2+shift_factor, 0],
                                       width=((2.5e9, 'cm'), (y_limit*1e5, 'cm')),
                                       origin=(0, 0, 'domain'), fontsize=32)
                    slc.set_cmap('gradient_magnitude_Te', 'copper')
                    slc.set_log('gradient_magnitude_Te', False)
                    slc.set_axes_unit("m")
                    ds.current_time = yt.units.yt_array.YTQuantity(ti*dt*second)
#                    slc.annotate_timestamp(redshift=False, draw_inset_box=True,coord_system='figure')
                    save_folder = root_save_dir+'t'+str(ti).zfill(4)+'/'
                    Path(save_folder).mkdir(parents=True, exist_ok=True)
                    slc.save(save_folder+'T_'+str(ti))
        # data frame to nest data in
        if data_save == True:
            del_rho_x_y = sch_plot_abs_grad(rho_data,np.transpose(x_grid0)[0],
                                            y_grid0[0])
            data.append(np.hstack([path_numerics, del_rho_x_y.max()]))
            df = pd.DataFrame(data, columns=['driver time [s]', 
                                            'magnetic field strength [G]',
                                            'amplitude [km s-1]',
                                            'max gradrho'],
                                            index = [i for i in range(np.shape(data)[0])])

if data_save == True:
    # save data
    df.to_pickle('max_delrho_dt.dat')

## merge pandas
#supernest = pd.concat([nest1, nest2])

## how to read pickels
#df_read_test = pd.read_pickle('test.dat')
## how to create the df in dfs 
#df_collect = pd.DataFrame({'idx':[[100,50,50],2,3], 'dfs':[dft1, dft2, dft3]})
## matches = test = [ind for ind, i in enumerate(df_collect['idx']) if sum(i-[200, 100, 80])==0]
# df_collect['dfs'][test[0]] 



