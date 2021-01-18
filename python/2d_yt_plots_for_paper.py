import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import img2vid as i2v
import glob
import yt
from yt.units import second
from pathlib import Path
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


SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
# 2D
#path_2_file = '/j/2D/P300/B50/A60/'
#file_name = 'jet_P300_B50A_60_'
#B
#path_2_file = '/j/B/P50/B60/A60/'
#file_name = 'jet_P50_B60A_60_'

#path_2_files = ['/j/B/P300/B60/A20/','/j/B/P300/B60/A80/','/j/B/P300/B20/A60/','/j/B/P300/B80/A60/','/j/B/P300/B60/A60/','/j/B/P200/B60/A60/','/j/B/P50/B60/A60/']
#file_names = ['jet_P300_B60A_20_','jet_P300_B60A_80_','jet_P300_B20A_60_','jet_P300_B80A_60_','jet_P300_B60A_60_','jet_P200_B60A_60_','jet_50_B60A_60_']

#path_2_files = ['/j/B/P50/B60/A60/']
#file_names = ['jet_P50_B60A_60_']

#path_2_files = ['/j/hight_dt/P300/B50/A60/']
#file_names = ['jet_P300_B50A_60_']

#path_2_files = ['/j/hdt/P300/B50/A40/']
#file_names = ['jet_P300_B50A_40_']

#path_2_files = ['/j/B/P300/B60/A60/']
#file_names = ['jet_P300_B60A_60_']

path_2_files = ['/j/B/P300/B60/A20/','/j/B/P300/B60/A80/','/j/B/P300/B20/A60/','/j/B/P300/B80/A60/','/j/B/P200/B60/A60/','/j/B/P50/B60/A60/']
file_names = ['jet_P300_B60A_20_','jet_P300_B60A_80_','jet_P300_B20A_60_','jet_P300_B80A_60_','jet_P200_B60A_60_','jet_P50_B60A_60_']

#Full_path = path_2_shared_drive + path_2_file + file_name

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
##highdt
#dt = unit_time/200

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

#otpions
save_fig =True
contour = False
dat_file = False
vtu_file = True
save_grid = False
B_plots = True

apex_and_width_pts = False
slice_height = 2e8
peak_hi = 0


if B_plots == True:
# mag tension is not okay at lower bc
    shift_x_factor = 2.4e2
else:
    shift_x_factor = 0

xres = 4096
yres = 2944
grid_size = np.divide(DOMIAN,[yres, xres])
physical_grid_size_xy =  DOMIAN/np.array([xres,yres])*cm_to_Mm

#var_names = ['temperature','magnetic_tension_x','density','pressure']
#cmaps = [ 'coolwarm','seismic', 'gist_heat', 'Greens']

#2D
#t0 = 0
#t1 = 81
#nb_steps = 21

##B
#t0 = 0
#t1 = 141
#nb_steps = 28

t0 = 0
t1 = 141
nb_steps = 142

#t0 = 1
#t1 = 2
#nb_steps = 1


##highdt
#t0 = 0
#t1 = 1140
#nb_steps = 300

## testing for side points
#t0 = 68
#t1 = 76
#nb_steps = 3

var_names = ['density']
cmaps = ['gist_heat']


#t0 = 8
#t1 = 8
#nb_steps = 1

time_stamps = np.linspace(t0, t1, nb_steps, dtype=int)
physical_time = time_stamps*dt
for stuff in range(len(file_names)):
    height_x = []
    height_y = []
    side_x = []
    side_y = []
    dis_y = []
    dis_x = []
    side_time = []

    FIRST = True
    file_name = file_names[stuff]
    path_2_file = path_2_files[stuff]
    Full_path = path_2_shared_drive + path_2_file + file_name

    for ind, ti in enumerate(time_stamps):
        jet_sides_index1_array = []
        jet_sides_index2_array = []
        offset = str(ti).zfill(4)
        save_folder = os.path.join(file_name,'t_'+offset)
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        if dat_file==True:
            ds = apt.load_datfile(Full_path+offset+'.dat')
            ad = ds.load_all_data()
            
            var_rho_data = ad['rho']*unit_density
            var_p_data = ad['p']*unit_pressure
            var_b1_data = ad['b1']*unit_magenticfield
            var_b2_data = ad['b2']*unit_magenticfield
    
        if vtu_file==True:
            ds = apt.load_vtkfile(ti, file=Full_path, type='vtu')
    
            var_rho_data = apt.vtkfiles.rgplot(ds.rho, data=ds, cmap='hot')
            var_rho_data, x_grid0, y_grid0 = var_rho_data.get_data(xres=xres, yres=yres)
    
            var_p_data = apt.vtkfiles.rgplot(ds.p, data=ds, cmap='hot')
            var_p_data, x_grid0, y_grid0 = var_p_data.get_data(xres=xres, yres=yres)
    
            var_b1_data = apt.vtkfiles.rgplot(ds.b1, data=ds, cmap='hot')
            var_b1_data, x_grid0, y_grid0 = var_b1_data.get_data(xres=xres, yres=yres)
            
            var_b2_data = apt.vtkfiles.rgplot(ds.b2, data=ds, cmap='hot')
            var_b2_data, x_grid0, y_grid0 = var_b2_data.get_data(xres=xres, yres=yres)
    
            var_T_data = apt.vtkfiles.rgplot(ds.T, data=ds, cmap='hot')
            var_T_data, x_grid0, y_grid0 = var_T_data.get_data(xres=xres, yres=yres)
    
            plt.close('all')
    
        arr_rho = np.zeros((var_rho_data.shape[0], var_rho_data.shape[1], 1))
        arr_p = np.zeros((var_p_data.shape[0], var_p_data.shape[1], 1))
        arr_b1 = np.zeros((var_b1_data.shape[0], var_b1_data.shape[1], 1))
        arr_b2 = np.zeros((var_b2_data.shape[0], var_b2_data.shape[1], 1))
        arr_T = np.zeros((var_T_data.shape[0], var_T_data.shape[1], 1))
        
        arr_rho[:, :, 0] = var_rho_data
        arr_p[:, :, 0] = var_p_data
        arr_b1[:, :, 0] = var_b1_data
        arr_b2[:, :, 0] = var_b2_data
        arr_T[:, :, 0] = var_T_data
        
        if apex_and_width_pts==True:
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
            thresh_hold = 15
            sorted_data = np.where(var_tr_data < 15, 0, 1)

            #dims in [y,x]
            shape = np.shape(sorted_data)
            if FIRST == True:
                if ti==0:
                    mid_pt_x = round(shape[0]/2)
                    clip_range_x = round(0.1*shape[0]) 
                    scan_range_x = [mid_pt_x-clip_range_x, mid_pt_x+clip_range_x]
    
                    mid_pt_y = round(shape[1]/2)
                    clip_range_y = round(0.2*shape[1]) 
                    scan_range_y = [0, mid_pt_y+clip_range_y]
                else:
                    #These don't work for first time step
                    indexs_x = np.nonzero(sorted_data[:,0])[0]
                    mid_pt_x = int(round((min(indexs_x)+(max(indexs_x)-min(indexs_x))/2)))
                    clip_range_x = round(0.1*shape[0]) 
                    scan_range_x = [mid_pt_x-clip_range_x, mid_pt_x+clip_range_x]
                    
                    mid_pt_y = round(shape[1]/2)
                    clip_range_y = round(0.2*shape[1]) 
                    scan_range_y = [0, mid_pt_y+clip_range_y]
                    x_extent = np.asarray(scan_range_x)*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm
                    cf = sum(x_extent)/2
                    x_extent -= cf 
                    y_extent = np.asarray(scan_range_y)*physical_grid_size_xy[1]
                    
                    FIRST = False
            if ti==0:     
                pass
            else:
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
                jet_sides_index1 = None
                # plot side values every 1 Mm interval
                for hi in range(1,int(np.floor(height_y[ind-1]))+1):
                    if hi>peak_hi: peak_hi=hi
                    slice_height = hi/cm_to_Mm
                    jet_sides_index1, jet_sides_index2, val1, val2 = side_pts_of_jet_dt(sorted_data, slice_height, DOMIAN, shape)
                    if jet_sides_index1==None:
                        pass
                    else:
                        dis_x.append((val2[0]-val1[0])*cm_to_km)
                        dis_y.append(slice_height)
    
                        side_x.append(val1[0])
                        side_x.append(val2[0])
                        side_y.append(slice_height)
                        side_y.append(slice_height)
                        side_time.append(ti*dt)
                        jet_sides_index1_array.append(jet_sides_index1)
                        jet_sides_index2_array.append(jet_sides_index2)

        data = dict(density=(arr_rho, "g/cm**3"), b1=(arr_b1, "gauss"),
                    b2=(arr_b2, "gauss"),pressure=(arr_p,"dyne/cm**2"),
                    temperature=(arr_T,"K"))
        bbox = np.array([[-2.5e4-shift_x_factor, 2.5e4-shift_x_factor], [0, 3e4], [-1, 1]])
        ds = yt.load_uniform_grid(data, arr_rho.shape, length_unit="km",
                                  bbox=bbox, nprocs=128)
    #    for i in sorted(ds.field_list):
    #        print(i)
        
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
            mag_tension_x = (b1*dxb1+b2*dyb1)/(4*np.pi)
            mag_tension_y = (b1*dxb2+b2*dyb2)    
            return mag_tension_x
            
        # trying to calc magnetic tension
        def _magnetic_tension_rescaled_(field, data):
            dxb1 = data['b1_gradient_x'] 
            dyb1 = data['b1_gradient_y']
            dxb2 = data['b2_gradient_x']
            dyb2 = data['b2_gradient_y']
            b1 =  data['b1']
            b2 =  data['b2']
            rescale_factor = 1e5
            mag_tension_x_rescaled = ((b1*dxb1+b2*dyb1)/(4*np.pi))*rescale_factor*10
            return mag_tension_x_rescaled
        
        # The gradient operator requires periodic boundaries.  This dataset has
        # open boundary conditions.  We need to hack it for now (this will be fixed
        # in future version of yt)
        ds.periodicity = (True, True, True)
        
        ds.add_field(('stream','magnetic_tension_x'), function=_magnetic_tension_, units="dyne/cm**3", take_log=False,
                     sampling_type="cell")
    
        ds.add_field(('stream','mag_tension_x_rescaled'), function=_magnetic_tension_rescaled_, units="Pa/m", take_log=False,
                     sampling_type="cell")    
        
        for idx, var_name in enumerate(var_names):
            # 2Mm
            y_limit = 0.8e4
            if var_name == 'magnetic_tension_x':
            # mag tension is not okay at lower bc
                shift_factor = 0.099e3
            else:
                shift_factor = 0
    
            slc = yt.SlicePlot(ds, "z", [var_name], center=[0.0, y_limit/2+shift_factor, 0],
                               width=((2.5e8, 'cm'), (y_limit*1e5, 'cm')),
                               origin=(0, 0, 'domain'), fontsize=52)

            ds.current_time = yt.units.yt_array.YTQuantity(ti*dt*second)
            slc.annotate_title('Time: '+str(np.round(ds.current_time.value, 2))+' '+str(ds.current_time.units))            
#            slc.annotate_timestamp(redshift=False, draw_inset_box=True,coord_system='figure')
            slc.set_cmap(var_name, cmaps[idx])
            slc.set_log(var_name, False)
#            slc.set_axes_unit("m")
            slc.set_axes_unit("Mm")
            
            if var_name == 'density' or var_name == 'rho':
                if contour is True:
                    slc.annotate_contour("mag_tension_x_rescaled",label=True, plot_args={"colors": "blue",
                                                   "linewidths": 4})#,  clim=(-200,200))
    
                slc.set_zlim('density', 4.5e-15*g_cm3_to_kg_m3,
                             1.5e-10*g_cm3_to_kg_m3)
                slc.set_unit(var_name, 'kg/m**3')
                ds.current_time = yt.units.yt_array.YTQuantity(ti*dt*second)
                #slc.annotate_timestamp(redshift=False, draw_inset_box=True,coord_system='figure')
                # need to add scatter plot here.
                if apex_and_width_pts==True:
                    if ti==0:
                        pass
                    else:
                        h_x = (jet_top_pixel_pos[0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf
                        h_y = jet_top_pixel_pos[1]*physical_grid_size_xy[1]
                        slc.annotate_marker([h_x, h_y], marker='^', coord_system='plot',
                                           plot_args={'color':'yellow', 's':1500})
                        if jet_sides_index1==None:
                            pass
                        else:                
                            # Overplot a feature at data location (x, y, z)
                            for nb_sides in range(len(jet_sides_index2_array)):
                                left_x = (jet_sides_index1_array[nb_sides][0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf
                                left_y = jet_sides_index1_array[nb_sides][1]*physical_grid_size_xy[1]
        
                                right_x = (jet_sides_index2_array[nb_sides][0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf
                                right_y = jet_sides_index2_array[nb_sides][1]*physical_grid_size_xy[1]
        
                                slc.annotate_marker([left_x, left_y], marker='o', coord_system='plot',
                                                    plot_args={'color':'blue', 's':800})                        
        
                                slc.annotate_marker([right_x, right_y], marker='o', coord_system='plot',
                                                    plot_args={'color':'blue', 's':800})                            
            if var_name == 'pressure':
                slc.set_zlim('pressure', 7.0e-1*dyne_cm2_to_Pa, 3e2*dyne_cm2_to_Pa)
                slc.set_unit(var_name, 'Pa')        
    
            if var_name == 'temperature':
                slc.set_zlim('temperature', 8e3,2e6)
                slc.set_unit(var_name, 'K')
    
            if var_name == 'magnetic_tension_x':
                if dat_file == True:
                    slc.set_unit(var_name, 'Pa/m')

    #                slc.set_zlim('mag_tension_x', -2e-5, 2e-5)
                else:
                    slc.set_unit(var_name, 'Pa/m')
            if save_fig == True:
                slc.save(save_folder+'/'+file_name+var_name+offset)
    
        if save_grid == True:
            ad = ds.all_data()
            arr = ad["magnetic_tension_x"].value 
            grid = ad.fcoords.value 
            indexs_sort_x = np.argsort(grid[:,0])
            sorted_arr = [grid[indexs_sort_x,0],
                          grid[indexs_sort_x,1],
                          arr[indexs_sort_x]]
            mag_T_arr = np.zeros((xres,yres))
            for x in range(xres):
                idy1 = x*yres
                idy2 = idy1+yres
                y_sort_ids = np.ndarray.tolist(np.argsort(sorted_arr[1][idy1:idy2])+idy1)
                mag_T_arr[x][:yres] = sorted_arr[2][y_sort_ids]*10 # 10 converts from cgs to si
      
    #            mag_T_arr = np.flip(np.transpose(mag_T_arr))
            # save pandas
            mid_x = int(xres/2)
            nb_grids_x = 100
            nb_grids_y = 1100        
            x_area = [int(mid_x-nb_grids_x),int(mid_x+nb_grids_x)]
           
            pd.DataFrame(mag_T_arr[x_area[0]:x_area[-1],0:nb_grids_y]).to_csv(save_folder+'/magnetic_tension_x_'+file_name+var_name+offset+'.csv',index=False)
            pd.DataFrame(var_rho_data[x_area[0]:x_area[-1],0:nb_grids_y]).to_csv(save_folder+'/rho'+file_name+var_name+offset+'.csv',index=False)
            pd.DataFrame(var_p_data[x_area[0]:x_area[-1],0:nb_grids_y]).to_csv(save_folder+'/pressure'+file_name+var_name+offset+'.csv',index=False)
            # save txt file
            np.savetxt(save_folder+'/magnetic_tension_x_'+file_name+var_name+offset+'.txt', mag_T_arr[x_area[0]:x_area[-1],0:nb_grids_y],)
            np.savetxt(save_folder+'/rho_'+file_name+var_name+offset+'.txt', var_rho_data[x_area[0]:x_area[-1],0:nb_grids_y])
            np.savetxt(save_folder+'/pressure_'+file_name+var_name+offset+'.txt', var_p_data[x_area[0]:x_area[-1],0:nb_grids_y])
    
    
    
