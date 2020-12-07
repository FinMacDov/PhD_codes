import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import img2vid as i2v
import glob
from skimage.measure import profile_line

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

from amrvac_pytools.datfiles.reading import amrvac_reader
from amrvac_pytools.vtkfiles import read, amrplot

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A40/'
file_name = 'jet_P300_B50A_40_'
Full_path = path_2_shared_drive + path_2_file + file_name



#plt.plot(z_data[0])

#plt.pcolormesh(x_grid, ygrid, z_data)
#plt.show()
# profile_line(2d_data,[0,0],[x1,y1],[x2,y2])
# verticle profile
#test = profile_line(z_data,[0,0],[0,4000], linewidth=1, order=0,  mode='nearest')
#horizobntal profile


def dis_2_grid(pt, res, array):
    # input physical distand and it will return its point on the grid
    # translate physical length to grid
    # shifts indexs if min pt inst 0 
    # pt - physical distance
    # res - how many grid points on line of interest
    # array - whole line in x or y direction
    dx = (abs(array[0])+abs(array[-1]))/res
    cf = - array[0]/dx
    return int(round(pt+cf))


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


yname = 'Distance [cm]'
xname = 'Time [s]'
cbar_label = 'v2 [cm s-1]'
UNIT_TIME = 8.58731
vmin = -4.5e6
vmax = 4.5e6
t0 = 0
tend = 160
step_size = 1

xres = 1000
yres = 8000

ds0 = read.load_vtkfile(0, file=Full_path, type='vtu')
data0 = amrplot.rgplot(ds0.v2, data=ds0, cmap='hot')
plt.close()
z_data0, x_grid0, y_grid0 = data0.get_data(xres=xres, yres=yres)
xpts = x_grid0[:, 0]
ypts = y_grid0[0, :]

# example to use function
index = dis_2_grid(2e6, xres, xpts)

xy0 = [index, 0]
xy1 = [index, yres-1]


t_step = (tend-t0)/step_size
time = np.linspace(t0, tend, t_step, dtype='int')
first = True
td_plot = []
for t in time:
    ds = read.load_vtkfile(t, file=Full_path, type='vtu')
    data = amrplot.rgplot(ds.v2, data=ds, cmap='hot')
    plt.close()
    z_data, x_grid, y_grid = data.get_data(xres=xres, yres=yres)
    if first is True:
        xpts = x_grid[:, 0]
        ypts = y_grid[0, :]
        first = False
        distance = np.sqrt((xpts[xy0[0]] - xpts[xy1[0]])**2 +
                           (ypts[xy0[-1]] - ypts[xy1[-1]])**2)
    slice_tn = profile_line(z_data, xy0, xy1, linewidth=1, order=0,
                            mode='nearest')
    td_plot.append(slice_tn)

plt.close()
# need to create a dist time grid
dis_array = np.linspace(0,distance,len(td_plot[0]))
time_grid, dis_grid = np.meshgrid(time*UNIT_TIME, dis_array)

##creat fig
fig = plt.subplot()

plt.xlabel(xname)
plt.ylabel(yname)

image = plt.pcolormesh(time_grid, dis_grid, np.transpose(td_plot),
                       cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
plt.colorbar(format='%.0e').set_label(cbar_label)
plt.show()

##plt.savefig('/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/sims/j/python/td_plot'+var+'.png')
