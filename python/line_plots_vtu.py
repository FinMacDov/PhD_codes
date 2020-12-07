import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import img2vid as i2v
import glob

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

from amrvac_pytools.datfiles.reading import amrvac_reader
from amrvac_pytools.vtkfiles import read, amrplot


test = False


if test is True: 
    path_2_file = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/sims/promRT/sj/2D/P300/B50'
    file_name = 'solar_jet_B_g_'
    offset = 35
    
    # testing td plot function:
    test_Array = amrplot.tdplot(var='T',filename=path_2_file+file_name, 
                                file_ext='vtu',x_pts=[0,0],y_pts=[0,3e9],
                                interpolations_pts=800,offsets=[1,10],step_size=2,
                                min_value = 8e3, max_value=1.8e6, cmap='coolwarm')
    #plt.imshow(test_Array)
    #plt.colorbar(orientation='vertical')
    plt.show()
path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A40/'
file_name = 'jet_P300_B50A_40_'
Full_path = path_2_shared_drive + path_2_file + file_name
print(Full_path)
var = 'v2'
t_range = [0,10]#80]
print(Full_path)
v2_min_value = -4e6
v2_max_value = -v2_min_value
data = amrplot.tdplot(var=var, filename=path_2_shared_drive+path_2_file+file_name,
                      file_ext='vtu', x_pts=[-1e7, 3e7], y_pts=[0, 0],
                      interpolations_pts=1600, offsets=t_range, step_size=1,
                      cmap='coolwarm',title='v2')
#                      min_value=v2_min_value, max_value=v2_max_value, cmap='coolwarm',title='v2')

plt.savefig('/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/sims/j/python/td_plot'+var+'.png')
