import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
#from PIL import Image
#import img2vid as i2v
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import pandas as pd

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

#from amrvac_pytools.datfiles.reading import amrvac_reader
#from amrvac_pytools.vtkfiles import read, amrplot

import amrvac_pytools as apt

def dis_2_grid(slice_height, physical_length, resoltion):
    # translate ypts height into index value
    convert = resoltion/physical_length
    return int(round(slice_height*convert))


def grid_2_dis(physical_length, resoltion, clip_domain, positon):
    # convert index position to physical length
    # physical_length of domain - [x,y]
    # resoultion of domian - [x,y]
    # clip_domain resolution of smaller regoin to returen correct x vbalue
    # positon of interest for conveting into units - [x,y]
    physical_length = np.asarray(physical_length)
    resoltion = np.asarray(resoltion)
    positon = np.asarray(positon)
    mid_pt_x = round(clip_domain[0]/2)
    # redefine zero poitn to jet centre
    positon[0] -= mid_pt_x
    convert = physical_length/resoltion
    return positon*convert

# function
def side_pts_of_jet_dt(clipped_data, slice_height, DOMIAN, shape):
    # gets the side points of the jet at spefified height for td plot
    # clipped_data - data set to scan 
    # slice_height - phyiscal value of where to take slice in x dir
    # DOMIAN - phyiscal length of whole domain (before clipping)
    # shape - shape of whole domain before clipping
    clip_domain = np.shape(clipped_data)
   
    # pick up the sides of jets
    xslice_idex = dis_2_grid(slice_height, DOMIAN[1], shape[1])

    x_slice = sorted_data[:, xslice_idex]

    if np.sum(x_slice) == 0:
        jet_sides_index1 = None
        jet_sides_index2 = None

        side_values1 = None
        side_values2 = None
    else:
        indexs_x = np.nonzero(x_slice)
    
        jet_sides_index1 = [min(indexs_x[0]), xslice_idex]
        jet_sides_index2 = [max(indexs_x[0]), xslice_idex]
    
        side_values1 = grid_2_dis(DOMIAN, shape, clip_domain, jet_sides_index1)
        side_values2 = grid_2_dis(DOMIAN, shape, clip_domain, jet_sides_index2)
    return jet_sides_index1, jet_sides_index2, side_values1, side_values2

# function
def side_pts(clipped_data, slice_height, DOMIAN, shape):
    # gets the side points of the jet at spefified height for td plot
    # clipped_data - data set to scan 
    # slice_height - index value of where to take slice in x dir
    # DOMIAN - phyiscal length of whole domain (before clipping)
    # shape - shape of whole domain before clipping
    clip_domain = np.shape(clipped_data)

    x_slice = sorted_data[:, slice_height]

    if np.sum(x_slice) == 0:
        jet_sides_index1 = None
        jet_sides_index2 = None

        side_values1 = None
        side_values2 = None
    else:
        indexs_x = np.nonzero(x_slice)

        jet_sides_index1 = [min(indexs_x[0]), slice_height]
        jet_sides_index2 = [max(indexs_x[0]), slice_height]

        side_values1 = grid_2_dis(DOMIAN, shape, clip_domain, jet_sides_index1)
        side_values2 = grid_2_dis(DOMIAN, shape, clip_domain, jet_sides_index2)
    return jet_sides_index1, jet_sides_index2, side_values1, side_values2


# function
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


path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A60/'
file_name = 'jet_P300_B50A_60_'

Full_path = path_2_shared_drive + path_2_file + file_name

dir_paths =  glob.glob('../2D/P*/B*/A*')


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

dt = unit_time/10

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

#otpions
testing = True

#t0 = 1
#t1 = 80
#nb_steps = 80

t0 =10
t1 = 20
nb_steps = 2


slice_height = 2e8

height_x = []
height_y = []

side_x = []
side_y = []
dis_y = []
dis_x = []
side_time = []

time_stamps = np.linspace(t0, t1, nb_steps, dtype=int)
physical_time = time_stamps*dt


xres = 4096
yres = 2944
data = []
for path in dir_paths:
    path_parts = path.split('/')
    path_parts = path_parts[len(path_parts)-3:]
    path_numerics = np.zeros(len(path_parts))
    for j, item in enumerate(path_parts):
        path_numerics[j] = float(item[1:])
    data.append(path_numerics)
df = pd.DataFrame(data, columns=['driver time [s]', 
                            'magnetic field strength [G]',
                            'amplitude [km s-1]'])

# save data

df.to_pickle('test.dat')

## merge pandas
#supernest = pd.concat([nest1, nest2])

## how to read pickels
#df_read_test = pd.read_pickle('test.dat')
## how to create the df in dfs 
#df_collect = pd.DataFrame({'idx':[[100,50,50],2,3], 'dfs':[dft1, dft2, dft3]})
## matches = [ind for ind, i in enumerate(df_collect['idx']) if i==[100, 50, 50]]

for ind, ti in enumerate(time_stamps):
    # Reading vtu file, allows to set custum grid poitns
    ds0 = apt.load_vtkfile(ti, file=Full_path, type='vtu')
    data0 = apt.vtkfiles.rgplot(ds0.trp1, data=ds0, cmap='hot')
    plt.close()
    var_tr_data, x_grid0, y_grid0 = data0.get_data(xres=xres, yres=yres)

    
    grad_tr = np.gradient(var_tr_data)
    grad_x = abs(grad_tr[0])
    grad_y = abs(grad_tr[1])
    # sum gradients togethers
    grad_total = grad_x+grad_y
    #create binary image
    thresh_hold = 10
    sorted_data = np.where(grad_total < thresh_hold, 0, 1)
    
    #dims in [y,x]
    shape = np.shape(sorted_data)
    
    mid_pt_x = round(shape[0]/2)
    clip_range_x = round(0.1*shape[0]) 
    scan_range_x = [mid_pt_x-clip_range_x, mid_pt_x+clip_range_x]
    
    mid_pt_y = round(shape[1]/2)
    clip_range_y = round(0.2*shape[1]) 
    scan_range_y = [0, mid_pt_y+clip_range_y]
    
    # clips data around jet
    sorted_data = sorted_data[scan_range_x[0]:scan_range_x[-1],
                              scan_range_y[0]:scan_range_y[-1]]
    
    # All indexs that belong to jet bc
    indexs = np.nonzero(sorted_data)
    
    # index for top of jet
    jet_top_index = np.argmax(indexs[1])
    jet_top_pixel_pos = [indexs[0][jet_top_index], indexs[1][jet_top_index]]
    # need to fix x postion as its zero point is not at jet centre
    values = grid_2_dis(DOMIAN, shape, np.shape(sorted_data), jet_top_pixel_pos)
    #top
    height_x.append(values[0]*cm_to_Mm)
    height_y.append(values[1]*cm_to_Mm)
    if testing == True:
        # testing
        cmap = 'gray'
        plt.scatter(jet_top_pixel_pos[1],jet_top_pixel_pos[0], s=30, color='red')
        # image
        plt.imshow(sorted_data, cmap=cmap)
        plt.colorbar()
        plt.savefig('image_check/image'+str(round(physical_time[ind]))+'.png', format='png', dpi=500)
        plt.clf()

#pickle.dump([physical_time, height_y], open('height_data.dat', 'wb'))
#pickle.dump([side_time, dis_x], open('width_data.dat', 'wb'))

plt.xlabel('Time [s]')
plt.ylabel('Height [Mm]')

plt.plot(physical_time, height_y, '-o', 
         color='red', linewidth=4,  markersize=6,
         markeredgecolor='black', markeredgewidth=1.5)
         
plt.savefig('image_check/test_hi.png', format='png', dpi=500)
plt.clf()
