from subplot_animation import subplot_animation
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

sys.path.append("/home/smp16fm/forked_amrvac/amrvac/tools/python")

from amrvac_pytools.datfiles.reading import amrvac_reader
from amrvac_pytools.vtkfiles import read, amrplot


program_name = sys.argv[0]
path2files = sys.argv[1:]

# Switches
refiner = '__'
fps = 3
start_frame = 0
in_extension = 'png'
out_extension = 'avi'
# set time to look over
time_start = 0
time_end = None
text_x_pos = 0.85
text_y_pos = 0.01
save_dir = '/shared/mhd_jet1/User/smp16fm/j/2D/results'
# make dirs
#path2files = "/shared/mhd_jet1/User/smp16fm/sj/2D/P300/B100/A20/"
#    path2files = "../test/"
#    dummy_name = 'solar_jet_con_'
dummy_name = ''

#read.load_vtkfile(0, file='/shared/mhd_jet1/User/smp16fm/sj/2D/P300/B100/A20/jet_t300_B100A_20_', type='vtu')

print(path2files[0])
test = subplot_animation(path2files[0],  save_dir=save_dir, dummy_name='',
                         refiner=None, text_x_pos=0.85, text_y_pos=0.01,
                         time_start=0, time_end=time_end, start_frame=0, fps=fps,
                         in_extension='png', out_extension='avi')

