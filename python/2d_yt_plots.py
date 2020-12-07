import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import img2vid as i2v
import glob
import yt

sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

#from amrvac_pytools.datfiles.reading import amrvac_reader
#from amrvac_pytools.vtkfiles import read, amrplot

import amrvac_pytools as apt

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A60/'
file_name = 'jet_P300_B50A_60_'
Full_path = path_2_shared_drive + path_2_file + file_name


ds = apt.load_datfile(Full_path+'0020.dat')
ad = ds.load_all_data()

unit_length = 1e9  # cm
unit_temperature = 1e6  # K
unit_numberdensity = 1e9  # cm^-3

unit_density = 2.3416704877999998E-015
unit_velocity = 11645084.295622544
unit_pressure = 0.31754922400000002
unit_magenticfield = 1.9976088799077159
unit_time = unit_length/unit_velocity

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

var_rho = 'rho'
var_b1 = 'b1'
var_b2 = 'b2'

contour = False

var_name = 'mag_tension_x' #'density'# 

cmap = 'seismic'# 'gist_heat' # 

var_rho_data = ad[var_rho]*unit_density
var_b1_data = ad[var_b1]*unit_magenticfield
var_b2_data = ad[var_b2]*unit_magenticfield

arr_rho = np.zeros((var_rho_data.shape[0], var_rho_data.shape[1], 1))
arr_b1 = np.zeros((var_b1_data.shape[0], var_b1_data.shape[1], 1))
arr_b2 = np.zeros((var_b2_data.shape[0], var_b2_data.shape[1], 1))

arr_rho[:, :, 0] = var_rho_data
arr_b1[:, :, 0] = var_b1_data
arr_b2[:, :, 0] = var_b2_data
print(var_rho_data[0],var_b2_data[0])

data = dict(density=(arr_rho, "g/cm**3"), b1=(arr_b1, "gauss"),
            b2=(arr_b2, "gauss"))
bbox = np.array([[-2.5e4, 2.5e4], [0, 3e4], [-1e4, 1e4]])
ds = yt.load_uniform_grid(data, arr_rho.shape, length_unit="km",
                          bbox=bbox, nprocs=128)
for i in sorted(ds.field_list):
    print(i)

# trying to use yt for magnetic tension. doesnt work
b1_grad_fields = ds.add_gradient_fields(('stream', 'b1'))
b2_grad_fields = ds.add_gradient_fields(('stream', 'b2'))          

# trying to calc magnetic tension
def _magnetic_tension_(field, data):
    dxb1 = data['b1_gradient_x'] 
    dyb1 = data['b1_gradient_y']
    dxb2 = data['b2_gradient_x']
    dyb2 = data['b2_gradient_y']
    b1 =  data['b1']
    b2 =  data['b2']
    mag_tension_x = b1*dxb1+b2*dyb1
    mag_tension_y = b1*dxb2+b2*dyb2    
    return mag_tension_x

# The gradient operator requires periodic boundaries.  This dataset has
# open boundary conditions.  We need to hack it for now (this will be fixed
# in future version of yt)
ds.periodicity = (True, True, True)

ds.add_field(('stream','mag_tension_x'), function=_magnetic_tension_, units="gauss**2/cm", take_log=False,
             sampling_type="cell")

## 2Mm
#y_limit = 1.99e4
#slc = yt.SlicePlot(ds, "z", [var_name], center=[0.0, -y_limit, 0],
#                   origin=(0, -3.1e4, 'domain'),
#                   width=((1e9, 'cm'), (2e9, 'cm')))


## works very flixciably
#y_limit = 0.5e4
#slc = yt.SlicePlot(ds, "z", [var_name], center=[0.0, y_limit/2, 0],
#                   width=((5e8, 'cm'), (y_limit*1e5, 'cm')),
#                   origin=(0, 0, 'domain'))


# 2Mm
y_limit = 0.8e4
# mag tension is not okay at lower bc
# need to remove
shift_factor = 0.099e3
slc = yt.SlicePlot(ds, "z", [var_name], center=[0.0, y_limit/2+shift_factor, 0],
                   width=((8e8, 'cm'), (y_limit*1e5, 'cm')),
                   origin=(0, 0, 'domain'))


if contour is True:
    slc.annotate_contour("mag_tension_x",label=True, plot_args={"colors": "blue",
                                   "linewidths": 4})#,  clim=(-200,200))

slc.set_cmap(var_name, cmap)
slc.set_log(var_name, False)
slc.set_axes_unit("m")

if var_name == 'density' or var_name == 'rho':
    #plot.set_unit(var, 'g/cm**3')
    slc.set_unit(var_name, 'kg/m**3')


#slc.set_zlim(var_name, -200, 200)
#slc.annotate_grids(cmap=None)
slc.save(file_name+var_name)
