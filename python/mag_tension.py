# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:28:13 2020

@author: fionnlagh
"""

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
#dir_paths =  glob.glob('../B/P*/B*/A*')
dir_paths =  glob.glob('../2D/P300/B50/A60')

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
testing = False
plotting_on = False
data_save = True
stop_height_condition = 1e8 #5e6 # 5 cells high
stop_indx_condition = 24 # ~50s which is min driver time
thresh_hold = 0.4

t0 =10
t1 = 20
nb_steps = 2


xres = 4096
yres = 2944

grad_size = np.divide(DOMIAN,[yres, xres])

peak_hi = 0

big_data = []
data = []
big_data_indexs = []
for path in dir_paths:
    path_parts = path.split('/')
    path_parts = path_parts[len(path_parts)-3:]
    path_numerics = np.zeros(len(path_parts))

    for j, item in enumerate(path_parts):
        path_numerics[j] = float(item[1:])

    full_paths = glob.glob(path+'/jet_'+path_parts[0]+'_'+path_parts[1]+'*.vtu')
    # skip first step as no value
    full_paths = full_paths[1:]
    full_paths = [full_paths[30]]
    physical_time = []

    for ind, path_element in enumerate(full_paths):
        Full_path = path_2_shared_drive + path_element[2:-8]
        # need to fix ti
        ti = int(path_element[-8:-4])           
        # Reading vtu file, allows to set custum grid poitns
        ds0 = apt.load_vtkfile(ti, file=Full_path, type='vtu')
        data_b1 = apt.vtkfiles.rgplot(ds0.b1, data=ds0, cmap='hot')
        plt.close()
        data_b2 = apt.vtkfiles.rgplot(ds0.b2, data=ds0, cmap='hot')
        plt.close()
        b1, x_grid0, y_grid0 = data_b1.get_data(xres=xres, yres=yres)
        b2, x_grid0, y_grid0 = data_b2.get_data(xres=xres, yres=yres)

        b1 = np.flip(np.transpose(b1))
        b2 = np.flip(np.transpose(b2))

        grad_b1 = np.gradient(b1,grad_size[0],grad_size[1],edge_order=1)
        grad_b2 = np.gradient(b2,grad_size[0],grad_size[1],edge_order=1)

        dxb1 = grad_b1[0]
        dyb1 = grad_b1[1]

        dxb2 = grad_b2[0]
        dyb2 = grad_b2[1]
 
        mag_tension_x = (b1*dxb1)+(b2*dyb1)
        mag_tension_y = b1*dxb2+b2*dyb2 
 
plt.imshow(mag_tension_x)
#plt.clim(-2,2)
plt.set_cmap('seismic')
plt.colorbar()
plt.show()