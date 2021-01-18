import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg') # revert above
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

sys.path.append("/home/smp16fm/amrvac_v_2_1/amrvac/tools/python")

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
    
def data_slice(data, xranges, yranges, yh_index):
    # will produce slices for dt plotting
    clipped_data = data[xranges[0]:xranges[-1],
                        yranges[0]:yranges[-1]]
    slice_data = clipped_data[:,yh_index]
    return slice_data


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


path_2_shared_drive = '/shared/mhd_jet1/User/smp16fm/j'    
dir_paths = [os.getenv("given_path")]
#dir_paths = ['/shared/mhd_jet1/User/smp16fm/j/B/P300/B60/A60']
print(dir_paths)
#dir_paths =  glob.glob(path_2_shared_drive+'/hdt/P*/B*/A*')
#dir_paths =  glob.glob('../B/P*/B*/A*')
#dir_paths =  glob.glob('../hight_dt/P*/B*/A*')
#testing
#dir_paths = [dir_paths[30]]

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

dt = unit_time/200

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

# otpions
testing = True

plotting_on = False

data_save = False
data_save_dir = path_2_shared_drive+'/python/hdt_alt/'
td_plotting = True
td_plot_root_folder =path_2_shared_drive+'/python/td_plots_data_sharc/'
td_file_name = 'data.csv'

stop_height_condition = 1e8 #5e6 # 5 cells high
stop_indx_condition = 120# 24 # ~50s which is min driver time
thresh_hold = 20 #0.4

#t0 = 1
#t1 = 80
#nb_steps = 80

t0 =10
t1 = 20
nb_steps = 2

#time_stamps = np.linspace(t0, t1, nb_steps, dtype=int)
#physical_time = time_stamps*dt

xres = 4096
yres = 2944

physical_grid_size_xy =  DOMIAN/np.array([xres,yres])*cm_to_Mm

peak_hi = 0
testc = 0
#for path in dir_paths:
#manual clipping
for path in dir_paths:
    FIRST = True
    h_check = 0
    path_parts = path.split('/')
    path_parts = path_parts[len(path_parts)-3:]
    path_numerics = np.zeros(len(path_parts))

    for j, item in enumerate(path_parts):
        path_numerics[j] = float(item[1:])

    full_paths = glob.glob(path+'/jet_'+path_parts[0]+'_'+path_parts[1]+'*.vtu')
    # skip first step as no value
    full_paths = full_paths[1:]
    # testing
#    full_paths = [full_paths[102]]
   
    sub_data_1 = []
    sub_data_2 = []
    physical_time = []

    for ind, path_element in enumerate(full_paths):
        Full_path = path_element[:-8]
        jet_name = Full_path.split('/')[-1]
        # need to fix ti
        ti = int(path_element[-8:-4])           
        # Reading vtu file, allows to set custum grid poitns
        ds0 = apt.load_vtkfile(ti, file=Full_path, type='vtu')
        data0 = apt.vtkfiles.rgplot(ds0.trp1, data=ds0, cmap='hot')
        plt.close()
        var_tr_data, x_grid0, y_grid0 = data0.get_data(xres=xres, yres=yres)
            
#        grad_tr = np.gradient(var_tr_data)
#        grad_x = abs(grad_tr[0])
#        grad_y = abs(grad_tr[1])
#        # sum gradients togethers
#        grad_total = grad_x+grad_y
#        #create binary image
##        sorted_data = np.where(grad_total < thresh_hold, 0, 1)
#       alt meth
        sorted_data = np.where(var_tr_data < 15, 0, 1)
#        grad_total = np.around(grad_total,decimals=1)
#        sorted_data = np.where(grad_total > 0, 1, 0)        
        #dims in [y,x]
        shape = np.shape(sorted_data)
        # This mid point doesnt corospond to jet centre
        if FIRST == True:
            #These don't work for first time step
            indexs_x = np.nonzero(sorted_data[:,0])[0]
            mid_pt_x = int(round((min(indexs_x)+(max(indexs_x)-min(indexs_x))/2)))
#            mid_pt_x = 2067 
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
        height_x = values[0]*cm_to_Mm
        height_y = values[1]*cm_to_Mm
        physical_time = ti*dt
        sub_data_1.append((physical_time, height_y))
        
        # stops the loop for creating data
        if (height_y < stop_height_condition*cm_to_Mm and ind>stop_indx_condition): 
            break

        # plot side values every 1 Mm interval
        for hi in range(1,int(np.floor(height_y))+1):
            if h_check-hi<0:
                td_first=True
                h_check=hi
            if hi>peak_hi: peak_hi=hi
            slice_height = hi/cm_to_Mm
            jet_sides_index1, jet_sides_index2, val1, val2 = side_pts_of_jet_dt(sorted_data, slice_height, DOMIAN, shape)
            if jet_sides_index1==None:
                pass
            else:
                dis_x = (val2[0]-val1[0])*cm_to_km
                dis_y = slice_height*cm_to_Mm
        
                side_xL = val1[0]*cm_to_km
                side_xR = val2[0]*cm_to_km
#                side_y.append(slice_height*cm_to_Mm)
#                side_y.append(slice_height*cm_to_Mm)
                side_time = ti*dt
                sub_data_2.append((side_time, dis_x, dis_y, 
                                 side_xL, side_xR))
                # This wont work for tilted jets or any asymetries, its a quick fix
                if testing == True:
                    plt.scatter((jet_sides_index1[0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,
                                 jet_sides_index1[1]*physical_grid_size_xy[1],
                                 s=40, color='blue')
                    plt.scatter((jet_sides_index2[0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,
                                 jet_sides_index2[1]*physical_grid_size_xy[1],
                                 s=40, color='blue')
                if td_plotting == True:
                    td_save_path = td_plot_root_folder+full_paths[ind].split('/')[-1][:-9]+'/'+str(hi)+'Mm'
                    Path(td_save_path).mkdir(parents=True, exist_ok=True)
                    height_km = dis_y
                    height_index = jet_sides_index2[1]

                    datarho = apt.vtkfiles.rgplot(ds0.rho, data=ds0, cmap='hot')
                    plt.close()
                    datarho, dummy_x, dummy_y = datarho.get_data(xres=xres, yres=yres)
                    rho_slice = data_slice(datarho, scan_range_x, scan_range_y, height_index)*g_cm3_to_kg_m3
                    datarho = []

                    dataTe = apt.vtkfiles.rgplot(ds0.T, data=ds0, cmap='hot')
                    plt.close()
                    dataTe, dummy_x, dummy_y = dataTe.get_data(xres=xres, yres=yres)
                    Te_slice = data_slice(dataTe, scan_range_x, scan_range_y, height_index)
                    dataTe =[]

                    dataVx = apt.vtkfiles.rgplot(ds0.v1, data=ds0, cmap='hot')
                    plt.close()
                    dataVx, dummy_x, dummy_y = dataVx.get_data(xres=xres, yres=yres)
                    Vx_slice = data_slice(dataVx, scan_range_x, scan_range_y, height_index)*cm_to_km
                    dataVx = []

                    dataVy = apt.vtkfiles.rgplot(ds0.v2, data=ds0, cmap='hot')
                    dataVy, dummy_x, dummy_y = dataVy.get_data(xres=xres, yres=yres)
                    plt.close()
                    Vy_slice = data_slice(dataVy, scan_range_x, scan_range_y, height_index)*cm_to_km
                    dataVy = []
                    
                    td_xvales = [scan_range_x[0]*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,scan_range_x[1]*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf]

                    td_xranges = np.linspace(td_xvales[0],td_xvales[1], len(Vy_slice))

                    df_td = pd.DataFrame(np.transpose(np.array([np.ones(len(Vy_slice))*ti*dt, td_xranges, rho_slice, Te_slice, Vx_slice, Vy_slice, dis_x*np.ones(len(Vy_slice))])), columns=['time [s]', 'x [Mm]','density [kg m-3]','Te [k]','vx [km s-1]', 'vy [km s-1]', 'width [km]'])
                    if td_first == True:
#                        print('writting')
                        df_td.to_csv(td_save_path+'/'+td_file_name, index = False, columns=['time [s]', 'x [Mm]', 'density [kg m-3]','Te [k]','vx [km s-1]', 'vy [km s-1]','width [km]'])
                        td_first = False
                    else:
                        df_td.to_csv(td_save_path+'/'+td_file_name, mode='a', index = False, header=None)
        if testing == True:
            # testing
            cmap = 'gray'
            plt.scatter((jet_top_pixel_pos[0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,jet_top_pixel_pos[1]*physical_grid_size_xy[1], marker='^', s=40, color='yellow')
            # image
#            plt.imshow(sorted_data, cmap=cmap)
            plt.imshow(np.rot90(var_tr_data[scan_range_x[0]:scan_range_x[-1], scan_range_y[0]:scan_range_y[-1]]), cmap=cmap, extent = [x_extent[0], x_extent[1], y_extent[0],y_extent[1]])
            plt.xlim(-1.0,1.0)
            plt.ylim(0,8)
            plt.xlabel('x (Mm)')
            plt.ylabel('y (Mm)')
#            plt.colorbar()
            plt.show()
            plt.savefig(path_2_shared_drive+'/python/image_check_sj/jet_P'+str(int(path_numerics[0]))+'_B' +
                        str(int(path_numerics[1])) +
                        'A_' + str(int(path_numerics[2])) +
                        'T_'+str(round(physical_time)) + '.png',
                        format='png', dpi=500)
            plt.clf()

    # data frame to nest data in
    df_sub1 = pd.DataFrame(sub_data_1, columns=['time [s]', 
                                'Height [Mm]'],
                                 index = [i for i in range(len(sub_data_1))])        

    df_sub2 = pd.DataFrame(sub_data_2,
                                 columns=['side time [s]', 'jet Width [km]',
                                'height [Mm]', 'jet side left [km]',
                                'jet side right [km]'], 
                                 index = [i for i in range(len(sub_data_2))])
                                 
    big_data = pd.concat([df_sub1, df_sub2], axis=1)
    big_data_indexs = path_numerics.astype(int)    # first data set
    data = np.hstack([path_numerics,max(sub_data_1, key=lambda x: float(x[1]))[1]])
    df = pd.DataFrame([data], columns=['driver time [s]', 
                                    'magnetic field strength [G]',
                                    'amplitude [km s-1]',
                                    'max height [Mm]'],
                                    index = [i for i in range(np.shape(data)[0])])
    df_bd = pd.DataFrame({'idx':[big_data_indexs], 'dfs':[big_data]})
    if data_save==True:
#        testc+=1
#        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1'+data_save_dir+jet_name, testc)
        Path(data_save_dir+jet_name).mkdir(parents=True, exist_ok=True)
        df.to_pickle(data_save_dir+jet_name+'/'+jet_name+'max_h.dat')
        df_bd.to_pickle(data_save_dir+jet_name+'/'+jet_name+'big_data.dat')

        

## merge pandas
#supernest = pd.concat([nest1, nest2])

## how to read pickels
#df_read_test = pd.read_pickle('test.dat')
## how to create the df in dfs 
#df_collect = pd.DataFrame({'idx':[[100,50,50],2,3], 'dfs':[dft1, dft2, dft3]})
## matches = test = [ind for ind, i in enumerate(df_collect['idx']) if sum(i-[200, 100, 80])==0]
# df_collect['dfs'][test[0]] 


#pickle.dump([physical_time, height_y], open('height_data.dat', 'wb'))
#pickle.dump([side_time, dis_x], open('width_data.dat', 'wb'))

#if plotting_on == True:
#    plt.xlabel('Time [s]')
#    plt.ylabel('Height [Mm]')
#    
#    plt.plot(physical_time, height_y, '-o', 
#             color='red', linewidth=4,  markersize=6,
#             markeredgecolor='black', markeredgewidth=1.5)
#             
#    plt.savefig('image_check/test_hi.png', format='png', dpi=500)
#    plt.clf()
#
#    for hi in range(1,peak_hi):
#    #    print(hi)
#        dumma_array_for_idxs = np.asarray(dis_y)
#        side_time = np.asarray(side_time)
#        dis_x = np.asarray(dis_x)
#        idx_side = np.where(dumma_array_for_idxs==(hi))
#    #    print(side_time, idx_side)
#        plt.plot(side_time[idx_side], dis_x[idx_side], '-o',
#                 linewidth=4,  markersize=6,
#                 markeredgecolor='black', markeredgewidth=1.5)
#    #                color='blue', linewidth=4,  markersize=6,
#    #                markeredgecolor='black', markeredgewidth=1.5)
#    plt.xlabel('Time [s]')
#    plt.ylabel('Jet width [km]')
#    
#    plt.savefig('image_check/test_si.png', format='png', dpi=500)
#    plt.clf()
