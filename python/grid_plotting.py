import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import img2vid as i2v
import glob
import matplotlib.ticker as ticker


sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

#from amrvac_pytools.datfiles.reading import amrvac_reader
#from amrvac_pytools.vtkfiles import read, amrplot

import amrvac_pytools as apt


path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A40/'
file_name = 'jet_P300_B50A_40_'
Full_path = path_2_shared_drive + path_2_file + file_name


SMALL_SIZE = 26
MEDIUM_SIZE = 28
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ds = apt.load_datfile(Full_path+'0000.dat')

p = ds.amrplot('T', draw_mesh=True, mesh_linewidth=1.25, mesh_color='white',
               mesh_linestyle='solid', mesh_opacity=1,  cmap='coolwarm')
#p.ax.set_xlim(-1,1)
#p.ax.set_ylim(0,1.5)

#xticks = p.ax.get_xticks()*10
#yticks = p.ax.get_yticks()*10
#p.ax.set_xticklabels(xticks)
#p.ax.set_yticklabels(yticks)

p.ax.set_xlabel('x [Mm]')
p.ax.set_ylabel('y [Mm]')
p.ax.set_title(None)
p.colorbar.set_label('Te [MK]')
#p.fig.tight_layout()
ds.show()
