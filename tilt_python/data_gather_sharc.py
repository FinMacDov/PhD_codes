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
import math
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

#sys.path.append("/home/fionnlagh/forked_amrvac/amrvac/tools/python")

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

    x_slice = clipped_data[:, xslice_idex]

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

    x_slice = clipped_data[:, slice_height]

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
    
    
def angle_cal(A,B):
    # A & B are (x_n,y_n)
    Y1 = max(A[1],B[1])
    Y2 = min(A[1],B[1])
    x1, x2 = A[0], B[0]
    m = A-B
    if m[1]/m[0] < 0:
        theta = -np.arccos((Y1-Y2)/(np.sqrt((x1-x2)**2+(Y1-Y2)**2)))+2*np.pi
    else:
        theta = np.arccos((Y1-Y2)/(np.sqrt((x1-x2)**2+(Y1-Y2)**2)))
    return theta

def func(m, x, c):
    return m*x + c


def LoBf(xy_data):
    # reutrns a linear line of best fit
    x =  xy_data[:,0]
    y =  xy_data[:,1]
    # best fit will only return single val, crashin angle func
    if min(x)-max(x) == 0:
        angle=0
    else:
        xline = np.linspace(min(x),max(x),100)
        popt, pcov = curve_fit(func, x, y)
        yline = func(xline, *popt)
        start_pt, end_pt = np.asarray((xline[0], yline[0])), np.asarray((xline[-1], yline[-1]))
        angle = angle_cal(start_pt, end_pt) 
    return angle
    

def vec_angle(A,B,C):
    a = B-A
    b = C-B
    unit_vec_A = a/np.linalg.norm(a)
    unit_vec_C = b/np.linalg.norm(b)
#    print(A,B,C, a, b, np.linalg.norm(a),np.linalg.norm(b))
    dot_product = np.dot(unit_vec_A, unit_vec_C)
    theta = np.arccos(dot_product)
#    print(dot_product, np.arccos(dot_product), math.degrees(np.arccos(dot_product)))
    return theta
    

def distance_cal(A, B):
    # A = (x1, y1)
    # B = (x2, y2)
#    print('NUMBERS!!')
#    print(A[0]-B[0], B[1]-B[1],(A[0]-B[0])**2+(B[1]-B[1])**2)
    return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)


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

#path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
path_2_shared_drive = '/shared/mhd_jet1/User/smp16fm/j'
code_root = path_2_shared_drive+'/tilt_python/sharc_run'

dir_paths = [os.getenv("given_path")]
jet_fname = os.getenv("given_jet_name")
# Happy with print test
print(dir_paths,jet_fname)

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
testing = True
plotting_on = False
data_save = False
# NOTE: if name already there it will append to file
max_h_data_fname = code_root + '/' + jet_fname + '/max_h_data_sj_p2.dat'
big_data_fname = code_root + '/'+jet_fname + '/big_data_set_sj_p2.dat'
Path(code_root+ '/'+jet_fname).mkdir(parents=True, exist_ok=True)
td_plotting = False
td_plot_root_folder = code_root+'/td_plots_data_sj/'
td_file_name = 'data.csv'
stop_height_condition = 1e8 #5e6 # 5 cells high
stop_indx_condition = 120# 24 # ~50s which is min driver time
thresh_hold = 20 #0.4
central_axis_tracking = True
c_data_root = code_root+'/c_data/'
central_axis_step_size = 0.1 # Mm
# abdandon methods
method_1 = False
method_2 = False
# chosen method
method_3 = True
method_4 = True
pts_of_influence = 3 # need to be moved but here for convenice
x_pad = 1/2 #Mm
y_pad = 0.75/2 # Mm
dummy_dis = 0

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

big_data = []
data = []
big_data_indexs = []
for path in dir_paths:
    FIRST = True
    data_c_first = True
    JL_data_first = True
    h_check = 0
    path_parts = path.split('/')
    path_parts = path_parts[-4:]
    path_numerics = np.zeros(len(path_parts))

    for j, item in enumerate(path_parts):
#        print(item)
        path_numerics[j] = float(item[1:])

    full_paths = glob.glob(path+'/jet_'+path_parts[0]+'_'+path_parts[1]+'_'+path_parts[2]+'_'+path_parts[3]+'_*.vtu')
    # skip first step as no value
    full_paths = full_paths[1:]
   
    sub_data_1 = []
    sub_data_2 = []
    physical_time = []

    for ind, path_element in enumerate(full_paths):
#        Full_path = path_2_shared_drive + path_element[:-8]
        Full_path = path_element[:-8]
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
        bin_data = np.where(var_tr_data < 15, 0, 1)
#        grad_total = np.around(grad_total,decimals=1)
#        sorted_data = np.where(grad_total > 0, 1, 0)        
        #dims in [y,x]
        shape = np.shape(bin_data)
        # This mid point doesnt corospond to jet centre
        if FIRST == True:
            #These don't work for first time step
            indexs_x = np.nonzero(bin_data[:,0])[0]
            mid_pt_x = int(round((min(indexs_x)+(max(indexs_x)-min(indexs_x))/2)))
#            mid_pt_x = 2067 
            clip_range_x = round(0.1*shape[0]) 
            scan_range_x = [mid_pt_x-clip_range_x, mid_pt_x+clip_range_x]
            
            mid_pt_y = round(shape[1]/2)
#            clip_range_y = round(0.2*shape[1]) 
            clip_range_y = round(0.2*shape[1]) 
            scan_range_y = [0, mid_pt_y+clip_range_y]
            x_extent = np.asarray(scan_range_x)*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm
            cf = sum(x_extent)/2
            x_extent -= cf 
            y_extent = np.asarray(scan_range_y)*physical_grid_size_xy[1]
            x_pad_dex_size = int(np.ceil(x_pad/physical_grid_size_xy[0]))
            y_pad_dex_size = int(np.ceil(y_pad//physical_grid_size_xy[0]))
            
            FIRST = False
        
        # clips data around jet
        sorted_data = bin_data[scan_range_x[0]:scan_range_x[-1],
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
#            print(hi, h_check, h_check-hi, h_check-hi<0)
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

                    df_td = pd.DataFrame(np.transpose(np.array([np.ones(len(Vy_slice))*ti*dt, td_xranges, rho_slice, Te_slice, Vx_slice, Vy_slice])), columns=['time [s]', 'x [Mm]','density [kg m-3]','Te [k]','vx [km s-1]', 'vy [km s-1]'])
                    if td_first == True:
#                        print('writting')
                        df_td.to_csv(td_save_path+'/'+td_file_name, index = False, columns=['time [s]', 'x [Mm]', 'density [kg m-3]','Te [k]','vx [km s-1]', 'vy [km s-1]'])
                        td_first = False
                    else:
                        df_td.to_csv(td_save_path+'/'+td_file_name, mode='a', index = False, header=None)

# test putting side tracking here
        # need to add more points to using above
        if central_axis_tracking == True:
            if height_y<1:
                pass
            else:
                nb_step = int((height_y/central_axis_step_size)+1) #+1 ensures endpts remain
                hi_locs = np.linspace(0,height_y,nb_step, endpoint=True)/cm_to_Mm

                central_pts = []
                central_sides = []
                for c_pts in hi_locs:
                    cjet_sides_index1, cjet_sides_index2, cval1, cval2 = side_pts_of_jet_dt(sorted_data, c_pts, DOMIAN, shape)
                    if cjet_sides_index1 is not None:
                        central_sides.append((cjet_sides_index1,cjet_sides_index2))
                        central_pts.append(np.add(cjet_sides_index1,cjet_sides_index2)//2)
                    else:
                        print('Cenrtal axis pt missed')
                        continue#                # add top position (not need due to hi_loc correction)
                # remember to remove
#                central_pts.append(np.asarray(jet_top_pixel_pos))
                central_pts = np.reshape(central_pts,np.shape(central_pts))
                if testing == True:
    #                        plt.plot(central_pts[:,0], central_pts[:,1])
                    plt.scatter((central_pts[:,0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,
                                    central_pts[:,1]*physical_grid_size_xy[1],
                                    s=40, color='yellow', marker="*", zorder=3)
                                    
                # need to calc length of jet
                central_pts_phy = np.zeros(np.shape(central_pts))
                central_pts_phy[:,:1] = (central_pts[:,:1]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf
                central_pts_phy[:,-1:] = central_pts[:,-1:]*physical_grid_size_xy[1]
                # vectorization of dis_Cal func
                p2p_dis = np.sqrt(np.sum((central_pts_phy[:-1]-central_pts_phy[1:])**2,axis=1))
                p2p_dis_array = np.zeros([np.size(p2p_dis),3])
                p2p_dis_array[:,1:] = central_pts_phy[1:]
                for i in range(1,len(p2p_dis)):
                    p2p_dis_array[i,0] = sum(p2p_dis[:i+1])
    
    #            jet_length = sum(p2p_dis)                
                jet_length = p2p_dis_array[-1][0] 
    #            print(jet_length)
                if data_save == True:
                    df_JL_data = pd.DataFrame([[jet_length, physical_time]],
                                              columns=['Jet length [Mm]',
                                                       'Time [s]'])
                    if JL_data_first:
                        data_c_save_path = c_data_root+full_paths[ind].split('/')[-1][:-9]
                        Path(data_c_save_path).mkdir(parents=True, exist_ok=True)
                        df_JL_data.to_csv(data_c_save_path+'/'+full_paths[ind].split('/')[-1][:-9]+'_'+'df_jl.csv', 
                                          index = False, columns=['Jet length [Mm]',
                                                                  'Time [s]'])
                        JL_data_first = False
                    else:
                        df_JL_data.to_csv(data_c_save_path+'/'+full_paths[ind].split('/')[-1][:-9]+'_'+'df_jl.csv', 
                                          mode='a', index = False, header=None)
                #-------------------------------------------
                if method_1 == True:
                    # trying method of avg angles
                    for hi_indx in range(1,len(central_pts)-1): 
                        p1,p2 = angle_cal(central_pts[hi_indx-1],central_pts[hi_indx]), angle_cal(central_pts[hi_indx],central_pts[hi_indx+1])
                        perp_avg_tilt = np.mean([p1,p2])-np.pi/2
                        m_grad = 1/np.tan(perp_avg_tilt)
                        const = central_pts[hi_indx][1]-m_grad*central_pts[hi_indx][0]
                        x_slit = np.linspace(0,clip_range_x*2,50)
                        line = m_grad*x_slit+const
                        if testing == True:
                            plt.plot((x_slit+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,line*physical_grid_size_xy[1], 'r-')
                # ------------------------------------------------------
                # method 2: middle angles
                if method_2 == True:
                    for hi_indx in range(1,len(central_pts)-1): 
                        vec_A = central_pts[hi_indx-1]
                        vec_B = central_pts[hi_indx]
                        vec_C = central_pts[hi_indx+1]
                        vec_ang = vec_angle(vec_A,vec_B,vec_C)
                        width_angle = vec_ang/2
                        m_grad = np.tan(width_angle)
                        const = vec_B[1]-m_grad*vec_B[0]
                        x_slit = np.linspace(0,clip_range_x*2,50)
                        line = m_grad*x_slit+const
                        if testing == True:
                            plt.plot((x_slit+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,line*physical_grid_size_xy[1], 'b--')

                if method_4 == True:
                    for hi_indx in range(1,int(np.floor(height_y))+1):
                        current_x_pad_dex_size = x_pad_dex_size
                        current_y_pad_dex_size = y_pad_dex_size
                        # +1 matches it with central_pts as 1 element is lost with dis calc
                        c_index = np.argmin(abs(p2p_dis_array[:,0]-hi_indx))+1
                        if (c_index+pts_of_influence >= len(p2p_dis_array)) and (c_index-pts_of_influence<0):
                            pass
                        else:
                            p1 = LoBf(central_pts[c_index-pts_of_influence:c_index+pts_of_influence])
                            perp_avg_tilt = p1-np.pi/2
                            m_grad = 1/np.tan(perp_avg_tilt)
        #                   current method
                            const = central_pts[c_index][1]-m_grad*central_pts[c_index][0]
                            z_line_switches = [0]
                            # makes sure that more than 1 edge is detected
                            while_count = 0
                            while sum(np.abs(z_line_switches)) < 2:
#                                print(while_count)
                                while_count += 1
                                # defines search region
                                x_search = (central_sides[c_index][0][0]-current_x_pad_dex_size,
                                            central_sides[c_index][1][0]+current_x_pad_dex_size)
                                y_search = (central_sides[c_index][0][1]-current_y_pad_dex_size,
                                            central_sides[c_index][0][1]+current_y_pad_dex_size)
                                # grid in phy units
                                points = np.array((y_grid0[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],y_search[0]:y_search[1]].flatten(),
                                                   x_grid0[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],y_search[0]:y_search[1]].flatten())).T*cm_to_Mm
                                values = (bin_data[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],
                                                      y_search[0]:y_search[1]]).flatten()

                                line_dis_phy = np.sqrt(((x_search[0]-x_search[-1])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf)**2+((y_search[0]-y_search[-1])*physical_grid_size_xy[1])**2)
                                nb_pts_for_line =  int(line_dis_phy//0.05)
                                x_slit = np.linspace(x_search[0],x_search[1],nb_pts_for_line)
                                x_slit_phy = (x_slit+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf

                                line = m_grad*x_slit+const
                                line_phy = line*physical_grid_size_xy[1]
                                xi = np.array(list(zip(line_phy, x_slit_phy)))

                                z_line_vale = griddata(points, values, xi)
                                z_line_vale[np.where(np.isnan(z_line_vale))]=0
                                z_line_vale = np.where(z_line_vale<1,0,1)
                                z_line_switches = np.diff(z_line_vale)
                                # expand search area                                
                                if sum(np.abs(z_line_switches)) < 2:
#                                    print('while not broken', sum(np.abs(z_line_switches)))
                                    current_x_pad_dex_size += 5 
                                    current_y_pad_dex_size += 5
                                    continue
#                                print('while will be broken', sum(np.abs(z_line_switches)))

                                # make sure only 2 pts are sleceted
                                LR_edge_fix = np.argwhere(abs(z_line_switches)>0)
                                LR_edge_fix_index = [np.min(LR_edge_fix),np.max(LR_edge_fix)]
                                spatial_locs_widths = xi[LR_edge_fix_index]
                                # Will be give Mm
                                tilt_widths = distance_cal(spatial_locs_widths[0],
                                                           spatial_locs_widths[1])

                                if testing == True:
                                    # Physical grid checking
                                    # Issue with grid aligment due to how yt written data, most likely cause by the sterech grids. 
                                    # width are correctly measure but are shift leftward due to difference in physical value for index pts of the grid and line
                                    extra_cf = (x_grid0[:,0][scan_range_x[0]+x_search[0]])*cm_to_Mm-min(x_slit_phy)
                                    plt.scatter(spatial_locs_widths[:,1:]-extra_cf,spatial_locs_widths[:,:-1], color='pink', marker='P', zorder=2)
                                    # test to purely size slice area
                                    plt.plot(x_slit_phy-extra_cf,line_phy, 'c:', zorder=1)
                # ------------------------------------------------------
                # method 3: top angles
                if method_3 == True:
                    for hi_indx in range(1,int(np.floor(height_y))+1):
                        current_x_pad_dex_size = x_pad_dex_size
                        current_y_pad_dex_size = y_pad_dex_size
                        # +1 matches it with central_pts as 1 element is lost with dis calc
                        c_index = np.argmin(abs(p2p_dis_array[:,0]-hi_indx))+1
                        # if value fall at top of arry angle cant be calc
                        if c_index+1 >= len(p2p_dis_array):
                            pass
                        else:
                            p1 = angle_cal(central_pts[c_index], central_pts[c_index+1])
                            perp_avg_tilt = p1-np.pi/2
                            m_grad = 1/np.tan(perp_avg_tilt)
                            # makes sure that more than 1 edge is detected
                            const = central_pts[c_index][1]-m_grad*central_pts[c_index][0]
                            z_line_switches = [0]
                            while_count = 0
        #                   current method
                            while sum(np.abs(z_line_switches)) < 2:
                                # defines search region
                                x_search = (central_sides[c_index][0][0]-current_x_pad_dex_size,
                                            central_sides[c_index][1][0]+current_x_pad_dex_size)
                                y_search = (central_sides[c_index][0][1]-current_y_pad_dex_size,
                                            central_sides[c_index][0][1]+current_y_pad_dex_size)
                                # grid in phy units
                                points = np.array((y_grid0[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],y_search[0]:y_search[1]].flatten(),
                                                   x_grid0[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],y_search[0]:y_search[1]].flatten())).T*cm_to_Mm
                                values = (bin_data[scan_range_x[0]+x_search[0]:scan_range_x[0]+x_search[1],
                                                      y_search[0]:y_search[1]]).flatten()
                                nb_pts_for_line =  int(line_dis_phy//0.05)
                                x_slit = np.linspace(x_search[0],x_search[1], nb_pts_for_line)
                                x_slit_phy = (x_slit+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf

                                line = m_grad*x_slit+const
                                line_phy = line*physical_grid_size_xy[1]
                                xi = np.array(list(zip(line_phy, x_slit_phy)))
     
                                z_line_vale = griddata(points, values, xi)
    #                            z_line_vale = z_line_vale[~np.isnan(z_line_vale)]
                                z_line_vale[np.where(np.isnan(z_line_vale))]=0
                                z_line_vale = np.where(z_line_vale<1,0,1)
                                z_line_switches = np.diff(z_line_vale)
                                if sum(np.abs(z_line_switches)) < 2:
#                                    print('while not broken', sum(np.abs(z_line_switches)))
                                    current_x_pad_dex_size += 5 
                                    current_y_pad_dex_size += 5
                                    continue
#                                print('while will be broken', sum(np.abs(z_line_switches)))                                
                                # make sure only 2 pts are sleceted
                                LR_edge_fix = np.argwhere(abs(z_line_switches)>0)
                                LR_edge_fix_index = [np.min(LR_edge_fix),np.max(LR_edge_fix)]
                                spatial_locs_widths = xi[LR_edge_fix_index]
                                # old method resultes in > 2pts
    #                            z_line_side_indexs = np.argwhere(abs(z_line_switches)>0)
    #                            spatial_locs_widths = xi[z_line_side_indexs][:,0]
    
    
                                # xi = [[y1,x1],[y2,x2]]
                                # Will be give Mm
                                tilt_widths = distance_cal(spatial_locs_widths[0],
                                                           spatial_locs_widths[1])
                                data_c = np.asarray((float(path_parts[0][1:]),
                                                     float(path_parts[1][1:]),
                                                     float(path_parts[2][1:]),
                                                     float(path_parts[3][1:]),
                                                     p2p_dis_array[c_index][0],
                                              p2p_dis_array[c_index][-1],
                                              tilt_widths, physical_time))
                                if data_save == True:
                                    df_dc = pd.DataFrame([data_c],
                                                         columns=['driver time [s]',
                                                                  'magnetic field strength [B]',
                                                                  'amplitude [km/s]',
                                                                  'tilt angle [degree]',
                                                                  'jet length [Mm]',
                                                                  'jet height [Mm]',
                                                                  'jet width [Mm]',
                                                                  'time [s]'])
                                    if data_c_first == True:
                #                        print('writting')
                                        data_c_save_path = c_data_root+full_paths[ind].split('/')[-1][:-9]
                                        Path(data_c_save_path).mkdir(parents=True, exist_ok=True)
                                        df_dc.to_csv(data_c_save_path+'/'+full_paths[ind].split('/')[-1][:-9]+'_'+td_file_name, 
                                                     index = False, columns=['driver time [s]',
                                                                             'magnetic field strength [B]',
                                                                             'amplitude [km/s]',
                                                                             'tilt angle [degree]',
                                                                             'jet length [Mm]',
                                                                             'jet height [Mm]',
                                                                             'jet width [Mm]',
                                                                             'time [s]'])
                                        data_c_first = False
                                    else:
                                        df_dc.to_csv(data_c_save_path+'/'+full_paths[ind].split('/')[-1][:-9]+'_'+td_file_name,
                                                     mode='a', index = False, header=None)                            
                                if testing == True:
                                    # Physical grid checking
        #                            cmap = 'gray'
                                    plt.plot(x_slit_phy,line_phy, 'g-', zorder=1)
                                    plt.scatter(spatial_locs_widths[:,1:],spatial_locs_widths[:,:-1], color='red', marker='s', zorder=2)
        #                            plt.plot((x_slit+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,line*physical_grid_size_xy[1], 'g-o')
        #                            plt.imshow(np.rot90(var_tr_data[scan_range_x[0]:scan_range_x[-1], scan_range_y[0]:scan_range_y[-1]]), cmap=cmap, extent = [x_extent[0], x_extent[1], y_extent[0],y_extent[1]])
        #                            plt.show()       

        if testing == True:
            # testing
            cmap = 'gray'
            plt.scatter((jet_top_pixel_pos[0]+scan_range_x[0])*physical_grid_size_xy[0]-2.547205e+09*cm_to_Mm-cf,jet_top_pixel_pos[1]*physical_grid_size_xy[1], s=40, color='red')
            # image
#            plt.imshow(sorted_data, cmap=cmap)
            plt.imshow(np.rot90(var_tr_data[scan_range_x[0]:scan_range_x[-1], scan_range_y[0]:scan_range_y[-1]]), cmap=cmap, extent = [x_extent[0], x_extent[1], y_extent[0],y_extent[1]])
            plt.xlim(-1,2.5)
            plt.ylim(0,8)
            plt.gca().set_aspect(0.5, adjustable='box')
            plt.xlabel('x (Mm)')
            plt.ylabel('y (Mm)')
#            plt.colorbar()
#            plt.show()
            #saves testing image
            Path(code_root+'/image_check/'+full_paths[ind].split('/')[-1][:-9]).mkdir(parents=True, exist_ok=True)
            plt.savefig(code_root+'/image_check/'+full_paths[ind].split('/')[-1][:-9]+'/jet_P'+str(int(path_numerics[0]))+'_B' +
                        str(int(path_numerics[1])) +
                        'A_' + str(int(path_numerics[2])) +
                        'T_'+str(ti).zfill(4) + '.png',
                        format='png', dpi=500)
            plt.clf()
    # data frame to nest data in
    df_sub1 = (pd.DataFrame(sub_data_1, columns=['time [s]', 
                                'Height [Mm]'],
                                 index = [i for i in range(len(sub_data_1))]))         

    df_sub2 = (pd.DataFrame(sub_data_2,
                                 columns=['side time [s]', 'jet Width [km]',
                                'height [Mm]', 'jet side left [km]',
                                'jet side right [km]'], 
                                 index = [i for i in range(len(sub_data_2))]))
                                 
#    big_data.append(pd.concat([df_sub1, df_sub2], axis=1))
#    big_data_indexs.append(path_numerics.astype(int))    # first data set
#    data.append(np.hstack([path_numerics,max(sub_data_1, key=lambda x: float(x[1]))[1]]))
#    df = pd.DataFrame(data, columns=['driver time [s]', 
#                                'magnetic field strength [G]',
#                                'amplitude [km s-1]',
#                                'Tilt [deg]',
#                                'max height [Mm]'],
#                                index = [i for i in range(np.shape(data)[0])])
    big_data = pd.concat([df_sub1, df_sub2], axis=1)
    big_data_indexs = path_numerics.astype(int)    # first data set
    df_collect = pd.DataFrame([{'idx':big_data_indexs, 'dfs':big_data}])

    data = np.hstack([path_numerics,max(sub_data_1, key=lambda x: float(x[1]))[1]])
    df = pd.DataFrame([data], columns=['driver time [s]', 
                                    'magnetic field strength [G]',
                                    'amplitude [km s-1]',
                                    'Tilt [deg]',
                                    'max height [Mm]'])
    if data_save == True:
    #    # save data
        if os.path.exists(max_h_data_fname):
        # add saving feature here
            data_max_h_t0 = pd.read_pickle(max_h_data_fname)
#            print('I add to file ' + max_h_data_fname + '!!!!!!!!!!' )
            dummy_max_h = data_max_h_t0.append(df, ignore_index=True)
            dummy_max_h.to_pickle(max_h_data_fname)
        else:
#            print('I made file ' + max_h_data_fname + '!!!!!!!!!!')
            df.to_pickle(max_h_data_fname)
    
    
        if os.path.exists(big_data_fname):
        # add saving feature here
            big_data_t0 = pd.read_pickle(big_data_fname)
#            print('I add to file ' + big_data_fname + '!!!!!!!!!!' )
            dummy_big_data = big_data_t0.append(df_collect, ignore_index=True)
            dummy_big_data.to_pickle(big_data_fname)
        else:
#            print('I made file ' + big_data_fname + '!!!!!!!!!!')
            df_collect.to_pickle(big_data_fname)

#df_collect = pd.DataFrame({'idx':big_data_indexs, 'dfs':big_data})
#if data_save == True:
#    # save data
#    df.to_pickle(max_h_data_fname)
#    df_collect.to_pickle('big_data_set_high_dt_p2.dat')

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
