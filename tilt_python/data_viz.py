import sys
import random as rd
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg') # revert above
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import glob
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pickle
import pandas as pd
from findiff import FinDiff
from scipy.stats import chisquare
from scipy.stats import spearmanr

def powlaw(x, a, b) :
    return np.power(10,a) * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata, sigma):
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    
    sigma_log = np.log10(sigma)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log,
                                   sigma=sigma_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)

def big_data_plotter(data_frame, x_name, y_name, index, ax, label, colour, style, lw, figsize):
    # plts big_data.data
    data_table = data_frame['dfs'][index]
#    print(y_name, x_name)
    return data_table.plot(ax=ax, kind='line', x=x_name, y=y_name, label=label,
                           c=colour, style=style, lw=lw, figsize=figsize)

def clipped_h_data_plotter(data_frame, index):
    # plts big_data.data
    h_data = data_frame['dfs'][index]['Height [Mm]'].dropna()
    x = h_data.index.values
    k = 3 # 5th degree spline
    n = len(h_data)
    s = 1#n - np.sqrt(2*n) # smoothing factor
    spline_1 = UnivariateSpline(x, h_data, k=k, s=s).derivative(n=1)
    sign_change_indx = np.where(np.diff(np.sign(spline_1(x))))[0] 
    if len(sign_change_indx)>1:
        sign_change_indx = sign_change_indx[1]
    else:
        sign_change_indx = len(h_data)
    return x[:sign_change_indx], h_data[:sign_change_indx]

def ballistic_flight(v0, g, t):
    # assumes perfectly verticle launch and are matching units
    # v0-initial velocity
    # g-gravitational acceleration
    # t-np time array
    x = v0*t
    y = v0*t-0.5*g*t**2
    y = np.where(y<0,0,y)

    t_apex = v0/g
    x_apex = v0*t_apex    
    y_apex = v0*t_apex-0.5*g*(t_apex)**2
    return x, y, t_apex, x_apex, y_apex

degree_sign= u'\N{DEGREE SIGN}'
i = 0
shuff = 0
SMALL_SIZE = 42
MEDIUM_SIZE = SMALL_SIZE + 2
BIGGER_SIZE = MEDIUM_SIZE + 2

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=26)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
path_2_shared_drive = '/run/user/1000/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
#dir_paths =  glob.glob('data/*')

##data set for paper
#dir_paths =  glob.glob('big_data/run1/*')
# run2 is standard jet runs 
#dir_paths =  glob.glob('big_data/run2/*')
# how to read pickels
#max_h_data_set = pd.read_pickle(dir_paths[1])
#big_data_set = pd.read_pickle(dir_paths[0])

#dir_paths_max_h =  glob.glob('sharc_run/jet_B60_A60_T*/max_h_data*')
#dir_paths_big_data = glob.glob('sharc_run/jet_B60_A60_T*/big_data*')

#ne
dir_paths_max_h =  glob.glob('sharc_run/new_pscan/*/max_h_data*')
dir_paths_big_data = glob.glob('sharc_run/new_pscan/*/big_data*')

#dir_paths_max_h =  glob.glob('sharc_run/tilt_scan/*/max_h_data*')
#dir_paths_big_data = glob.glob('sharc_run/tilt_scan/*/big_data*')


#dir_paths_max_h =  glob.glob('sharc_run/sj_tilt_Scan/*/max_h_data*')
#dir_paths_big_data = glob.glob('sharc_run/sj_tilt_Scan/*/big_data*')

dummy_max_h0 = []
max_h_data_set = []
dummy_bd0 = []
big_data_set = []

dummy_max_h0 = pd.read_pickle(dir_paths_max_h[0])
dummy_bd0 = pd.read_pickle(dir_paths_big_data[0])

# Silly fix for badly saved data.
# data files are being appended each run, need to fix this in other script
keep_indx = 1
if len(dummy_max_h0)>1:
#    dummy_max_h0 = dummy_max_h0.drop([0])
    for clip_idx in dummy_max_h0.index:
        if clip_idx == keep_indx:
            pass
        else:
            dummy_max_h0 = dummy_max_h0.drop([clip_idx])
if len(dummy_bd0)>1:
#    dummy_bd0 = dummy_bd0.drop([0])
    for clip_idx in dummy_bd0.index:
        if clip_idx == keep_indx:
            pass
        else:
            dummy_bd0 = dummy_bd0.drop([clip_idx])

first_append = True
for i in range(1,len(dir_paths_max_h)):
    dummy_max_h = pd.read_pickle(dir_paths_max_h[i])
    dummy_bd = pd.read_pickle(dir_paths_big_data[i])
    if len(dummy_max_h)>1:
        dummy_max_h = dummy_max_h.drop([0])
        for clip_idx in dummy_max_h.index:
            if clip_idx == keep_indx:
                pass
            else:
                dummy_max_h = dummy_max_h.drop([clip_idx])
    if len(dummy_bd)>1:
#        dummy_bd = dummy_bd.drop([0])
        for clip_idx in dummy_bd.index:
            if clip_idx == keep_indx:
                pass
            else:
                dummy_bd = dummy_bd.drop([clip_idx])
    if first_append == True:
        first_append = False
        max_h_data_set = dummy_max_h0.append(dummy_max_h,ignore_index=True)
        big_data_set = dummy_bd0.append(dummy_bd,ignore_index=True)
    else:
        max_h_data_set = max_h_data_set.append(dummy_max_h,ignore_index=True)
        big_data_set = big_data_set.append(dummy_bd,ignore_index=True)

max_h_data_set = max_h_data_set.sort_values(by=['Tilt [deg]'])#.reset_index(drop=True)
order = max_h_data_set.index
max_h_data_set = max_h_data_set.reset_index(drop=True)
big_data_set = big_data_set.reindex(order).reset_index(drop=True)
# constants
unit_length = 1e9  # cm
DOMIAN = [5*unit_length, 3*unit_length]
unit_temperature = 1e6  # K
unit_numberdensity = 1e9  # cm^-3

g_cm3_to_kg_m3 = 1e3
dyne_cm2_to_Pa = 1e-1
cm_to_km = 1e-5
m_to_km = 1e-3
km_to_Mm = 1e-3
cm_to_Mm = 1e-8
s_to_min = 1/60

earth_g = 9.80665  #m s-2
sun_g = 28.02*earth_g*m_to_km # km s-2

unit_density = 2.3416704877999998E-015
unit_velocity = 11645084.295622544
unit_pressure = 0.31754922400000002
unit_magenticfield = 1.9976088799077159
unit_time = unit_length/unit_velocity


# I messed up time scaling on data collection
TIME_CORRECTION_FACTOR = 10/unit_time

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

# options
# IMPORTANT TO CHANGE dt
dt = unit_time/20
#dt = unit_time/200 # high dt
plot_h_vs_t = True
all_data = True # plotss all data as suppose to small selection 
plot_w_vs_t = False
plot_error_bars = False
plot_hmax_vs_B = False #
plot_hmax_vs_A = False #

plot_mean_w_vs_tilt = True

power_law_fit = False
plot_hmax_vs_dt = False
data_check = False
interp_check = False # Doesnt work well enough for my purposes
diff_check = False
sf = [0.55, 0.55, 0.5, 0.5]
plot_mean_w_vs_BAdt = False
test_balstic = False
Decelleration_analysis = False
c_data = True
jet_word_search = 'jet_P300_B60_A60_T*/*data.csv'
jl_jet_word_search = 'jet_P300_B60_A60_T*/*jl.csv'
apex_vs_tile = True
plot_cdata_LA = True

quad_plot = True
quad_plot_cdata = True

lw =  3# 2.5#  
xliml11, xlimu11 = 15, 125
yliml11, ylimu11 = 1, 9

#xliml21, xlimu21 = 
yliml21, ylimu21 = 1, 12

#xliml21, xlimu21 = 
#yliml21, ylimu21 = 

#xliml22, xlimu22 = 
yliml22, ylimu22 = 100, 1800

# max_h_data_set.plot(x ='amplitude [km s-1]', y='max height [Mm]', kind = 'scatter')
# test = [ind for ind, i in enumerate(big_data_set['idx']) if sum(i-[50, 60, 20])==0]
#print(big_data_set['dfs'][test[0]])

#name = "tab20c"
#cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
#colors = cmap.colors  # type: list

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
styles = ['-', '--', '-.', ':','-', '--', '-.', ':','-']
styles_alt = ['-', '--', '-.', ':']

#default_cycler = (cycler(color=colors) +
#                  cycler(linestyle=styles))
#plt.rc('axes', prop_cycle=default_cycler)
plt.rc('lines', linewidth=lw)

#list_of_indexs = [[300,20,80],[200,20,80],[50,20,80],[300,80,80],[300,100,80]]

#list_of_indexs = [[20],[40],[60],[80]]

list_of_indexs = [[300,40]]

fig_len_h_comp=plt.figure(figsize=(60,40))                                                                                                                                                                                   
gs=GridSpec(2,2) # 2 rows, 2 columns
lhc_ax1 = fig_len_h_comp.add_subplot(gs[:,0])                                                                                                                                                                    
lhc_ax2 = fig_len_h_comp.add_subplot(gs[0,1])                                                                                                                                                                    
lhc_ax3 = fig_len_h_comp.add_subplot(gs[1,1], sharex=lhc_ax2)

if diff_check == True:
     id_no = 42
     driver_time = big_data_set['idx'][id_no][0]
     h_data = big_data_set['dfs'][id_no]['Height [Mm]'].dropna()
     t_data = big_data_set['dfs'][id_no]['time [s]'].dropna()
     time_stop_index = np.argmin(abs(t_data-driver_time))
     x = h_data.index.values
     dx = x[1] - x[0]
     
     d_dx = FinDiff(0, dx)     
     d2_dx2 = FinDiff(0, dx, 2)
          
     dh_dx = d_dx(h_data)
     d2h_dx2 = d2_dx2(h_data)

     mean = d2h_dx2[:time_stop_index].mean()
     std = d2h_dx2[:time_stop_index].std()
     sigma = 1

     range_of_vales = [mean-sigma*std,mean+sigma*std]
     test = d2h_dx2-mean
     step = np.hstack((np.ones(len(test)), -1*np.ones(len(test))))
     dary_step = np.convolve(test, step, mode='valid')
     step_indx = np.argmax(dary_step)
     
     clip_indx = np.argwhere((d2h_dx2>range_of_vales[0]) & (d2h_dx2<range_of_vales[1]))
     clip_indx = clip_indx.reshape(len(clip_indx))
     clip_data = h_data[clip_indx]     
     
     print(mean,std,range_of_vales)

     fig = plt.figure()
     ax = fig.add_subplot(111)
     
     ax.plot(t_data, h_data, 'bo', ms=2, label='data')
     ax.plot(t_data, dh_dx, 'r', label='1st order derivative')
#     ax.plot(t_data[:time_stop_index], d2h_dx2[:time_stop_index], 'b', label='2nd order derivative clip')  
#     ax.plot(t_data[clip_indx], d2h_dx2[clip_indx], 'g--', label='2nd order derivative')  
#     ax.plot(t_data[clip_indx], clip_data[clip_indx], 'orange', label='new curve')   
     ax.legend(loc='best')
     plt.show()
    

if interp_check == True:
#     id_no = 22
     id_no = 0
     h_data = big_data_set['dfs'][id_no]['Height [Mm]'].dropna()
     x = h_data.index.values
     k = 3 # 5th degree spline
     n = len(h_data)

     s = 1#n - np.sqrt(2*n) # smoothing factor
     spline_0 = UnivariateSpline(x, h_data, k=k, s=s)
     spline_1 = UnivariateSpline(x, h_data, k=k, s=s).derivative(n=1)
#     spline_2 = UnivariateSpline(x, h_data, k=k, s=s).derivative(n=2)
     sign_change_indx = np.where(np.diff(np.sign(spline_1(x))))[0] 
     if len(sign_change_indx)>1:
         sign_change_indx = sign_change_indx[1]
     else:
         sign_change_indx = len(h_data)
     
     fig = plt.figure()
     ax = fig.add_subplot(111)
    
     ax.plot(h_data, 'bo', ms=2, label='data')
     ax.plot(x, spline_0(x), 'k', label='5th deg spline')
#     ax.plot(x, spline_1(x), 'r', label='1st order derivative')
     ax.plot(x[:sign_change_indx], h_data[:sign_change_indx])
#     ax.plot(x, spline_2(x), 'g', label='2nd order derivative')
     ax.legend(loc='best')
     plt.show()

spear_list = []
decell_array = []
vmax_array = []
predicted_decell_array = []

if plot_cdata_LA:
    fig, ax = plt.subplots(figsize=(20,12))
    path2_c_data = glob.glob('sharc_run/c_data/'+jl_jet_word_search)
    tilt_nb = [int(path2_c_data[i].split('_')[-3][1:]) for i in range(len(path2_c_data))]
    tilt_order_index = np.argsort(tilt_nb)
    path2_c_data = [path2_c_data[i] for i in tilt_order_index]
    tilt_nb = [tilt_nb[i] for i in tilt_order_index]
    jet_length_max = []
    tilt_deg = []
    i=0
    for cdex, cdata_name in enumerate(path2_c_data):
        i = cdex % len(colors)
        j = cdex % len(styles_alt)
#        if i ==0: styles_alt = [styles_alt[-1]]+styles_alt[:-1]
        dumb_file = pd.read_csv(cdata_name)
        ax.plot(dumb_file['Time [s]'], dumb_file['Jet length [Mm]'],
                label=r'$\theta=$'+str(tilt_nb[cdex])+degree_sign, 
                color=colors[i], linestyle=styles_alt[j], linewidth=lw)
        lhc_ax3.plot(dumb_file['Time [s]'], dumb_file['Jet length [Mm]'],
                label=r'$\theta=$'+str(tilt_nb[cdex])+degree_sign, 
                color=colors[i], linestyle=styles_alt[j], linewidth=lw)
        jet_length_max.append(dumb_file['Jet length [Mm]'].max())
        i += 1        

    ax.legend(fontsize=24)
    ax.set_ylim(0,8) 
    plt.xlabel('Time [s]', fontsize=32)
    plt.ylabel('Length [Mm]', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    fig.savefig('sharc_run/fig_for_paper/'+'JL_vs_t_fixing.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(20,12))
    ax.plot(tilt_nb, jet_length_max, '-k', marker='o', markersize=20,
                linewidth=5)
    lhc_ax1.plot(tilt_nb, jet_length_max, '--r', marker='o', markersize=20,
                linewidth=5, label='Max Length')
    plt.xlabel('Tilt [' + degree_sign + ']', fontsize=32)
    plt.ylabel('Max length [Mm]', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    ax.set_ylim(4,8) 
    fig.savefig('sharc_run/fig_for_paper/'+'Tdeg_vs_JL_max_fixng'+'.png')
    plt.close()
        

if plot_h_vs_t == True:
    i=0
    if all_data == True:
        fig, ax = plt.subplots()
        for idx in range(len(big_data_set)):
            i = idx % len(colors)
            j = idx % len(styles_alt)
            big_data_plotter(big_data_set, x_name='time [s]', y_name='Height [Mm]',
                             index=idx, ax=ax, label=r'$\theta=$'+str(int(big_data_set['idx'][idx][-1]))+degree_sign,
                             colour=colors[i], style=styles_alt[j], lw=lw,
                             figsize=(20,12))
            big_data_plotter(big_data_set, x_name='time [s]', y_name='Height [Mm]',
                 index=idx, ax=lhc_ax2, label=r'$\theta=$'+str(int(big_data_set['idx'][idx][-1]))+degree_sign,
                 colour=colors[i], style=styles_alt[j], lw=lw,
                 figsize=(20,12))
        # Some reason pandas not plotting 
        ax.legend(fontsize=24)
        plt.xlabel('Time [s]', fontsize=32)
        plt.ylabel('Height [Mm]', fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.tick_params(axis='both', which='minor', labelsize=28)
        fig.savefig('sharc_run/fig_for_paper/'+'time_vs_height_fixing'+'.png')
        plt.close()
    else:
        for idx_num, idx_name in enumerate(list_of_indexs):
            fig, ax = plt.subplots(figsize=(20,12))
            #ax = plt.gca()
            idx_loc = [iloc for iloc, eye in enumerate(big_data_set['idx']) if sum((np.asarray([eye[0],eye[-2]])-np.asarray(idx_name)))==0]
#            idx_loc = [iloc for iloc, eye in enumerate(big_data_set['idx']) if sum([eye[0],eye[-2]]-idx_name))==0]
#            idx_loc = [7] # fudge to make code run single case
            if data_check == True:
                for iloc, eye in enumerate(big_data_set['idx']):
                    print(iloc,eye[0],idx_name,eye-idx_name,
                          sum(eye-idx_name),sum(abs(eye[-2]-idx_name))==0)
            t_max = 0
            for idx in idx_loc:
                i += 1
                i = i % len(colors)
                j = i % len(styles_alt)
                if i ==0: styles_alt = [styles_alt[-1]]+styles_alt[:-1]
                lab = 'P=' + str(big_data_set['idx'][idx][0])+ \
                ', B='+str(big_data_set['idx'][idx][1])+ \
                ', A='+str(big_data_set['idx'][idx][2])+ \
                r', $\theta$='+str(big_data_set['idx'][idx][3])
                      
#                big_data_plotter(big_data_set, x_name='time [s]',
#                                 y_name='Height [Mm]', index=idx, ax=ax, 
#                                 label=lab, colour=colors[i], style=styles_alt[j],
#                                 lw=lw, figsize=(20,12))                         
                data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
                data_x = data_x*dt+data_x[1]*dt #rescaling time and off by one time due to indexing
                # Decelleration cals
                if Decelleration_analysis==True:
                    grad_vert_check = np.where(np.gradient(data_y)>0,0,1)
                    if sum(grad_vert_check)>0:
                        max_h_index = data_y.argmax()
                        v_max = abs(np.gradient(data_y)/(dt*km_to_Mm)).max()
                        
                        v_rise_line = np.diff([data_y[0],data_y[max_h_index]])/(dt*max_h_index*km_to_Mm) # km/s-1
                        v_fall_line = np.diff([data_y[max_h_index], data_y[len(data_y)-1]])/(dt*(len(data_y)-max_h_index)*km_to_Mm) # km/s-1

                        v_rise = np.diff(data_y[:max_h_index])/(dt*km_to_Mm) # km/s-1
                        v_acell = abs(np.diff(v_rise)/dt) #km/s-1
                        
                        v_fall = np.diff(data_y[max_h_index:])/(dt*km_to_Mm) # km/s-1
                        v_decell = abs(np.diff(v_fall)/dt) #km/s-1
                        # from Heggland et al (2007)
                        predicted_decell = v_max/(0.5*big_data_set['idx'][idx][0]*m_to_km)

                        decell_array.append(v_decell.mean()/m_to_km) # m s-2
                        vmax_array.append(v_max)
                        predicted_decell_array.append(predicted_decell)
                    

                if test_balstic == True:
                    # Ballistic anaylsis                
                    ballstic_hor, ballistic_vert, t_apex, x_apex, y_apex = ballistic_flight(big_data_set['idx'][idx][2], sun_g, data_x)
                    ballistic_vert = ballistic_vert*km_to_Mm
                    if ballistic_vert[-1]==0:
                        zero_pt = np.where(ballistic_vert==0)[0][0]
                    else:
                        zero_pt = len(ballistic_vert)         
                    # https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/  
                    corr, _ = spearmanr(data_y[:zero_pt],ballistic_vert[:zero_pt])
                    spear_list.append([lab, corr])

                    test_fig, test_ax = plt.subplots(figsize=(20,12))
                    plt.plot(data_x,ballistic_vert)
                    plt.plot(data_x,data_y)
                    test_ax.set_ylabel("maximum height [Mm]")
                    test_ax.set_xlabel("time [s]")
                    test_ax.set_title(lab+' corr= '+str(corr))
                    test_fig.savefig('ballstic_test/P'+str(big_data_set['idx'][idx][0])+ \
                                '_B'+str(big_data_set['idx'][idx][1])+ \
                                '_A'+str(big_data_set['idx'][idx][2])+'test.png')
                    plt.close()
#                data_x = data_x*TIME_CORRECTION_FACTOR*dt
                if t_max < max(data_x): t_max = max(data_x)  
                ax.plot(data_x, data_y, color=colors[i], linestyle=styles_alt[j],
                         lw=lw+3, label=lab)
                #for ballstic test, goes here
            ax.legend(loc=1,fontsize=24)
            ax.set_ylabel("Maximum height [Mm]", fontsize=32)
            ax.set_xlabel("Time [s]", fontsize=32)
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
            # doesnt work for pandas
#            manager = plt.get_current_fig_manager()
#            manager.resize(*manager.window.maxsize())

            plt.gca().set_xlim(right=t_max+t_max*sf[idx_num])
            plt.savefig('P'+str(big_data_set['idx'][idx][0])+ \
                        '_B'+str(big_data_set['idx'][idx][1])+ \
                        '_A'+str(big_data_set['idx'][idx][2])+'_BF.png')
            plt.close()
        if Decelleration_analysis==True:
        # theorectical lines
            vmax_scan = np.linspace(min(vmax_array),max(vmax_array),500)
            predicted_decel_p50 = vmax_scan/(0.5*50*m_to_km)
            predicted_decel_p200 = vmax_scan/(0.5*200*m_to_km)
            predicted_decel_p300 = vmax_scan/(0.5*300*m_to_km)

            test_fig, test_ax = plt.subplots(figsize=(20,12))
            plt.scatter(decell_array, vmax_array,
                        label='Simulation data', color='red')
#            plt.scatter(predicted_decell_array, vmax_array, 
#                        marker = 'o', label='theoretical data')
            plt.plot(predicted_decel_p50,vmax_scan, label='P=50 s')
            plt.plot(predicted_decel_p200,vmax_scan, label='P=200 s')
            plt.plot(predicted_decel_p300,vmax_scan, label='P=300 s')
            test_ax.legend(loc=4, fontsize=24)
            test_ax.set_ylabel("maximum V [km s-1]", fontsize=32)
            test_ax.set_xlabel("decelerations [m s-2]", fontsize=32)
            test_ax.tick_params(axis='both', which='major', labelsize=28)
            test_ax.tick_params(axis='both', which='minor', labelsize=28)
            plt.show()
        # doesnt work for pandas

if apex_vs_tile == True:
    test_fig, test_ax = plt.subplots(figsize=(20,12))
    apex_data = []
    tilt_val = []
    for idx in range(len(big_data_set['idx'])):
        data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
        apex_data.append(data_y.max())
        tilt_val.append(big_data_set['idx'][idx][-1])
    plt.plot(tilt_val,apex_data,'-k', marker='o', markersize=20,
             linewidth=5)
    lhc_ax1.plot(tilt_val,apex_data,'-k', marker='o', markersize=20, linewidth=5, label='Max Height')
    test_ax.set_ylabel("Apex height [Mm]", fontsize=32)
    test_ax.set_xlabel("Tilt ["+degree_sign+"]", fontsize=32)
    test_ax.tick_params(axis='both', which='major', labelsize=28)
    test_ax.tick_params(axis='both', which='minor', labelsize=28)

    test_fig.savefig('sharc_run/fig_for_paper/apex_vs_tilt_fixing.png')
    plt.close()

data_mean_w = []
if plot_w_vs_t == True:
    i = 0
    t_max = 0
    w_root_dir = 'sharc_run/width_graphs'
    if not os.path.exists(w_root_dir):
        os.makedirs(w_root_dir)
    for idx in range(len(big_data_set)):
        fig, ax = plt.subplots(figsize=(20, 12))
        w_path = w_root_dir + '/P'+str(big_data_set['idx'][idx][0]) + \
                              '/B' + str(big_data_set['idx'][idx][1]) + \
                              '/A' + str(big_data_set['idx'][idx][2]) + \
                              '/T' + str(big_data_set['idx'][idx][3])
        if not os.path.exists(w_path):
            os.makedirs(w_path)
        lab = 'P=' + str(big_data_set['idx'][idx][0]) + \
              ', B=' + str(big_data_set['idx'][idx][1]) + \
              ', A=' + str(big_data_set['idx'][idx][2]) + \
              ', '+ r'$\theta=$' + str(big_data_set['idx'][idx][3])
        data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
        data_y = []
        t_max = max(data_x)*TIME_CORRECTION_FACTOR*dt
        jet_width = np.asanyarray(big_data_set['dfs'][idx]['jet Width [km]'])
        jet_time = np.asanyarray(big_data_set['dfs'][idx]['side time [s]'])*TIME_CORRECTION_FACTOR*dt
        height_markers = np.asanyarray(big_data_set['dfs'][idx]['height [Mm]'])
#        if t_max < max(data_x): t_max = max(data_x)
        if np.isnan(sum(big_data_set['dfs'][idx]['height [Mm]'])) == True:
            pass
        else:
            data_mean_w.append([big_data_set['idx'][idx][0],
                                big_data_set['idx'][idx][1],
                                big_data_set['idx'][idx][2],
                                big_data_set['idx'][idx][3],
                                big_data_set['dfs'][idx]['jet Width [km]'].mean()])
            for hi in range(1, int(max(height_markers))):
                i += 1
                i = i % len(colors)
                j = i % len(styles_alt)
                if i == 0: styles_alt = [styles_alt[-1]] + styles_alt[:-1]
                h_index = np.argwhere(height_markers == hi)
                ax.plot(jet_time[h_index], jet_width[h_index], color=colors[i],
                        linestyle=styles_alt[j],
                        lw=lw+3, label='height='+str(hi)+' Mm')
            ax.legend(loc=1, fontsize=24)
            ax.set_title(lab, fontsize=25)
            ax.set_ylabel("Jet width [km]", fontsize=32)
            ax.set_xlabel("Time [s]", fontsize=32)
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
            plt.savefig(w_path +'/P'+str(big_data_set['idx'][idx][0]) +
                        '_B' + str(big_data_set['idx'][idx][1]) +
                        '_A' + str(big_data_set['idx'][idx][2]) + '_BF.png')
        plt.close()
    df_w = pd.DataFrame(data_mean_w, columns=['driver time [s]',
                                              'magnetic field strength [G]',
                                              'amplitude [km s-1]',
                                              'tilt [degrees]',
                                              'mean width [km]'])
    # need to add new function that plots based on tilt. 
    # This should be for the old method of tilt measurements, not acounting for tilt of jet
    if plot_mean_w_vs_tilt == True:
        wmean_root_dir = w_root_dir+'/width_mean_graphs'
        if not os.path.exists(wmean_root_dir):
            os.makedirs(wmean_root_dir)
        i = 0
        j = 0
        # need ti write code that looks into the new method
        # data lies here: /tilt_python/c_data
        fig, ax = plt.subplots(figsize=(20, 12))
        if c_data == True:
            path2_c_data = glob.glob('sharc_run/c_data/'+jet_word_search)
            tilt_nb = [int(path2_c_data[i].split('_')[-2][1:]) for i in range(len(path2_c_data))]
            tilt_order_index = np.argsort(tilt_nb)
            path2_c_data = [path2_c_data[i] for i in tilt_order_index]
            c_width = []
            tilt_deg = []
            for cdex, cdata_name in enumerate(path2_c_data):
                dumb_file = pd.read_csv(cdata_name)
                c_width.append(dumb_file['jet width [Mm]'].mean()/km_to_Mm)
                tilt_deg.append(dumb_file['tilt angle [degree]'][0])
            plt.plot(tilt_deg,c_width,
                     label = 'Traced slit',
                     linestyle='-', marker='o', color = 'r',
                     markersize=20, linewidth=5)
        ax = df_w.plot(ax=ax, kind='line',
                          x='tilt [degrees]',
                          y='mean width [km]',
                          linestyle='--', marker='o', color='b',
                          label = 'Horizontal slit', markersize=20,
                          linewidth=5)
        plt.legend(loc='upper left', fontsize=24)
        ax.set_ylabel("Mean width [km]", fontsize=28)
        ax.set_xlabel('Tilt' + ' [' + degree_sign + ']', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.tick_params(axis='both', which='minor', labelsize=28)
#        plt.gca().set_xlim(right=115)
        plt.savefig(wmean_root_dir+'/mean_w_vs_tilt.png')
        plt.close()

    if plot_mean_w_vs_BAdt == True:
        wmean_root_dir = w_root_dir+'/width_mean_graphs'
        if not os.path.exists(wmean_root_dir):
            os.makedirs(wmean_root_dir)
        i = 0
        fig, ax = plt.subplots(figsize=(20, 12))
        for key_s, grp_s in df_w.groupby(['tilt [degrees]']):
            ground_nb = len(grp_s.groupby('driver time [s]').groups)
            ii = i*ground_nb % (len(colors)-1)
            i += 1
            shuff = 0
            for key, grp in grp_s.groupby(['amplitude [km s-1]']):
                for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                    j = shuff % len(styles)
                    shuff += 1
                    ax = sub_grp.plot(ax=ax, kind='line',
                                      x='magnetic field strength [G]',
                                      y='mean width [km]',
                                      label='P='+str(int(sub_key)) +
                                            ', A='+str(int(key)) + ', '+r'$\theta=$'+str(int(key_s)),
                                      c=colors[ii], style=styles[j])
                    if data_check == True:
                        print(sub_grp)
        plt.legend(loc=1, fontsize=24)
        ax.set_ylabel("Mean width [km]", fontsize=28)
        ax.set_xlabel('Magnetic field strength [G]', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.tick_params(axis='both', which='minor', labelsize=28)
#        plt.gca().set_xlim(right=115)
        plt.savefig(wmean_root_dir+'/mean_w_vs_B_BF.png')
        plt.close()

        i = 0
        fig, ax = plt.subplots(figsize=(20,12))
        for key_s, grp_s in df_w.groupby(['tilt [degrees]']):
            ground_nb = len(grp_s.groupby('driver time [s]').groups)
            ii = i*ground_nb % (len(colors)-1)
            i += 1
            shuff = 0            
            for key, grp in grp_s.groupby(['magnetic field strength [G]']):
                for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                    j = shuff % len(styles)
                    shuff += 1          
                    ax = sub_grp.plot(ax=ax, kind='line', x='amplitude [km s-1]', 
                                      y='mean width [km]',
                                      label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_s)),
                                      c=colors[ii], style=styles[j], lw=lw)
                    if data_check == True:
                        print(sub_grp)
        plt.legend(loc=1, fontsize=24)
        ax.set_ylabel("mean width [km]", fontsize=28)
        ax.set_xlabel('amplitude [km s-1]', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.tick_params(axis='both', which='minor', labelsize=28)
        ax.set_xlim(right=92)
        plt.savefig(wmean_root_dir+'/mean_w_vs_A_BF.png')
        plt.close()

#        fig, ax = plt.subplots(figsize=(20,12))
#        for key, grp in df_w.groupby(['magnetic field strength [G]']):
#            ground_nb = len(grp.groupby('driver time [s]').groups)
#            ii = i*ground_nb % (len(colors)-1)
#            i += 1
#            shuff = 0 
#            for sub_key, sub_grp in grp.groupby(['amplitude [km s-1]']):
#                j = shuff % len(styles)
#                shuff += 1
#                ax = sub_grp.plot(ax=ax, kind='line', x='driver time [s]', 
#                                  y='mean width [km]',
#                                  label='B='+str(int(key))+', A='+str(int(sub_key)),
#                                  c=colors[ii], style=styles[j], lw=lw)
#                if data_check == True:
#                    print(sub_grp)
##        plt.legend(bbox_to_anchor=(1.2,0.5), loc=7, fontsize=20)
#        plt.legend(loc=1, fontsize=24)
##        plt.tight_layout()
#        ax.set_ylabel("mean width [km]", fontsize=28)
#        ax.set_xlabel('driver time [s]', fontsize=28)
#        ax.tick_params(axis='both', which='major', labelsize=28)
#        ax.tick_params(axis='both', which='minor', labelsize=28)
#        ax.set_xlim(right=385)
#        plt.savefig(wmean_root_dir+'/mean_w_vs_P_BF.png')
#        plt.close()



if plot_hmax_vs_B == True:
    i=0
    fig, ax = plt.subplots(figsize=(20,12))
    for key_t, grp_t in max_h_data_set.groupby(['Tilt [deg]']):
        ground_nb_t = len(grp_t.groupby('driver time [s]').groups)
        ii = i*ground_nb_t % (len(colors)-1)
        i += 1
        shuff = 0
        for key, grp in grp_t.groupby(['amplitude [km s-1]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                j = shuff % len(styles)
                shuff += 1
                ax = sub_grp.plot(ax=ax, kind='line',
                                  x='magnetic field strength [G]', 
                                  y='max height [Mm]',
                                  label='P='+str(int(sub_key))+', A='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j])
                if data_check == True:
                    print(sub_grp)
    plt.gca().set_xlim(right=120)
    plt.legend(loc=1, fontsize=18)
    ax.set_ylabel("max height [Mm]", fontsize=28)
    ax.set_xlabel('magnetic field strength [G]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    plt.show()

if plot_hmax_vs_A == True:
    i = 0
    fig, ax = plt.subplots(figsize=(20,12))
    for key_t, grp_t in max_h_data_set.groupby(['Tilt [deg]']):
        ground_nb = len(grp_t.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 1
        shuff = 0        
        for key, grp in grp_t.groupby(['magnetic field strength [G]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                j = shuff % len(styles)
                shuff += 1          
                ax = sub_grp.plot(ax=ax, kind='line', x='amplitude [km s-1]', 
                                  y='max height [Mm]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
        plt.legend(loc=2)
        ax.set_ylabel("max height [Mm]")
        if power_law_fit == True:
            mean_h = []
            std_h = []
            v = []
            for key, grp in max_h_data_set.groupby(['amplitude [km s-1]']):
                 mean_h.append(grp['max height [Mm]'].mean())
                 std_h.append(grp['max height [Mm]'].std())
                 v.append(key)
            if plot_error_bars == True:
                plt.errorbar(v, mean_h, color='k', yerr=std_h, zorder=20, fmt='-o',
                             errorevery=1, barsabove=True, capsize=5, capthick=2)
            popt_log, pcov_log, ydatafit_log = curve_fit_log(v, mean_h, std_h)
            perr = np.sqrt(np.diag(pcov_log))
            nstd = 1 # to draw 5-sigma intervals
            popt_up = popt_log+nstd*perr
            popt_dw = popt_log-nstd*perr
            v = np.asarray(v)
    
            fit_up = np.power(10, linlaw(np.log10(v),*popt_up))
            fit_dw = np.power(10, linlaw(np.log10(v),*popt_dw))
            plt.plot(v,ydatafit_log,'crimson',linewidth=lw, marker='o',
                     markersize=12, label='average')
            ax.fill_between(v, fit_up, fit_dw, alpha=.25)
            test = powlaw(v, *popt_log) 
    plt.legend(fontsize=18)
    ax.set_ylabel("max height [Mm]", fontsize=28)
    ax.set_xlabel('amplitude [km s-1]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)


    plt.show()

if plot_hmax_vs_dt == True:
    fig, ax = plt.subplots(figsize=(20,12))
    for key, grp in max_h_data_set.groupby(['magnetic field strength [G]']):
        ground_nb = len(grp.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 1
        shuff = 0 
        for sub_key, sub_grp in grp.groupby(['amplitude [km s-1]']):
            j = shuff % len(styles)
            shuff += 1
            ax = sub_grp.plot(ax=ax, kind='line', x='driver time [s]', 
                              y='max height [Mm]',
                              label='B='+str(int(key))+', A='+str(int(sub_key)),
                              c=colors[ii], style=styles[j], lw=lw, loglog=False)
            if data_check == True:
                print(sub_grp)

    ax.set_xlim(right=365)
    plt.legend(loc=1, fontsize=18)
    ax.set_ylabel("max height [Mm]", fontsize=28)
    ax.set_xlabel('driver time [s]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    plt.show()
    plt.savefig('hmax_vs_dt.png')
    plt.close()
    
if quad_plot == True:
    lw =  6
    multi_fig, ((ax11,ax12), (ax21, ax22)) = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(30,20))
    i = 0
    max_h_data_set = max_h_data_set.sort_values('Tilt [deg]',
                                        ascending=False)

    for key_t, grp_t in max_h_data_set.groupby(['Tilt [deg]'], sort=False):
        ground_nb = len(grp_t.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0        
        for key, grp in grp_t.groupby(['magnetic field strength [G]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['amplitude [km s-1]'])
                j = shuff % len(styles)
                shuff += 1          
                plot_pd = sub_grp.plot(ax=ax21, kind='line', x='amplitude [km s-1]', 
                                  y='max height [Mm]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
#        ax21.legend(loc=2)
#        ax21.set_ylabel("Max height [Mm]")
#        ax21.set_xlabel("A [km s-1]")
        if power_law_fit == True:
            mean_h = []
            std_h = []
            v = []
            for key, grp in max_h_data_set.groupby(['amplitude [km s-1]']):
                 mean_h.append(grp['max height [Mm]'].mean())
                 std_h.append(grp['max height [Mm]'].std())
                 v.append(key)
            if plot_error_bars == True:
                ax21.errorbar(v, mean_h, color='k', yerr=std_h, zorder=20, fmt='-o',
                             errorevery=1, barsabove=True, capsize=5, capthick=2)
            popt_log, pcov_log, ydatafit_log = curve_fit_log(v, mean_h, std_h)
            perr = np.sqrt(np.diag(pcov_log))
            nstd = 1 # to draw 5-sigma intervals
            popt_up = popt_log+nstd*perr
            popt_dw = popt_log-nstd*perr
            v = np.asarray(v)
    
            fit_up = np.power(10, linlaw(np.log10(v),*popt_up))
            fit_dw = np.power(10, linlaw(np.log10(v),*popt_dw))
            ax21.plot(v,ydatafit_log,'crimson',linewidth=lw, marker='o',
                     markersize=12, label='average')
            ax21.fill_between(v, fit_up, fit_dw, alpha=.25)
            test = powlaw(v, *popt_log) 
    ax21.legend(loc=2)
    ax21.set_ylim(yliml21,ylimu21)
    ax21.set_ylabel("Max height [Mm]")
    ax21.set_xlabel('A [km s-1]')
    ax21.tick_params(axis='both', which='major')
    ax21.tick_params(axis='both', which='minor')
    i = 0
    for key_t, grp_t in max_h_data_set.groupby(['Tilt [deg]'], sort=False):
        ground_nb_t = len(grp_t.groupby('driver time [s]').groups)
        ii = i*ground_nb_t % (len(colors)-1)
        i += 5
        shuff = 0
        for key, grp in grp_t.groupby(['amplitude [km s-1]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['magnetic field strength [G]'])
                j = shuff % len(styles)
                shuff += 1
                plot_pd = sub_grp.plot(ax=ax11, kind='line',
                                  x='magnetic field strength [G]', 
                                  y='max height [Mm]',
                                  label='P='+str(int(sub_key))+', A='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
    ax11.set_xlim(xliml11,xlimu11)
    ax11.set_ylim(yliml11,ylimu11)
    ax11.legend(loc=1)
    ax11.set_ylabel("Max height [Mm]")
    ax11.set_xlabel('B [G]')
    ax11.tick_params(axis='both', which='major')
    ax11.tick_params(axis='both', which='minor')

    t_max = 0
    for idx in range(len(big_data_set)):
        data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
        data_y = []
        t_max = max(data_x)*TIME_CORRECTION_FACTOR*dt
        jet_width = np.asanyarray(big_data_set['dfs'][idx]['jet Width [km]'])
        jet_time = np.asanyarray(big_data_set['dfs'][idx]['side time [s]'])*TIME_CORRECTION_FACTOR*dt
        height_markers = np.asanyarray(big_data_set['dfs'][idx]['height [Mm]'])
#        if t_max < max(data_x): t_max = max(data_x)
#        nan_cleaner_dex = np.argwhere(np.isnan([big_data_set['dfs'][idx]['height [Mm]']])==False)[:,-1]
        # not sure why I used this, it seems to remove whole data sets that I need
        # maybe past me is smarter and had a reason for this
#        if np.isnan(sum(big_data_set['dfs'][idx]['height [Mm]'])) == True:
#            print(idx)
#            pass
#        else:
#            data_mean_w.append([big_data_set['idx'][idx][0],
#                                big_data_set['idx'][idx][1],
#                                big_data_set['idx'][idx][2],
#                                big_data_set['idx'][idx][3],
#                                big_data_set['dfs'][idx]['jet Width [km]'].mean()])

        data_mean_w.append([big_data_set['idx'][idx][0],
                            big_data_set['idx'][idx][1],
                            big_data_set['idx'][idx][2],
                            big_data_set['idx'][idx][3],
                            big_data_set['dfs'][idx]['jet Width [km]'].mean()])

    df_w = pd.DataFrame(data_mean_w, columns=['driver time [s]',
                                              'magnetic field strength [G]',
                                              'amplitude [km s-1]',
                                              'tilt [degrees]',
                                              'mean width [km]'])
    df_w = df_w.sort_values('tilt [degrees]',
                                        ascending=False)
    i = 0
    for key_s, grp_s in df_w.groupby(['tilt [degrees]'], sort=False):
        ground_nb = len(grp_s.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0
        for key, grp in grp_s.groupby(['amplitude [km s-1]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['magnetic field strength [G]'])
                j = shuff % len(styles)
                shuff += 1
                plot_pd = sub_grp.plot(ax=ax12, kind='line',
                                  x='magnetic field strength [G]',
                                  y='mean width [km]',
                                  label='P='+str(int(sub_key)) +
                                        ', A='+str(int(key)) + ', '+r'$\theta=$'+str(int(key_s)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
    ax12.legend(loc=1)
    ax12.set_ylabel("Mean width [km]")
    ax12.set_xlabel('B [G]')
    ax12.tick_params(axis='both', which='major')
    ax12.tick_params(axis='both', which='minor')

    i = 0
    for key_s, grp_s in df_w.groupby(['tilt [degrees]'], sort=False):
        ground_nb = len(grp_s.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0            
        for key, grp in grp_s.groupby(['magnetic field strength [G]']):
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['amplitude [km s-1]'])
                j = shuff % len(styles)
                shuff += 1          
                plot_pd = sub_grp.plot(ax=ax22, kind='line', x='amplitude [km s-1]', 
                                  y='mean width [km]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_s)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
#    ax22.legend(loc=1)
    ax22.set_ylim(yliml22,ylimu22)
    ax22.legend(loc=2, ncol=2)
    ax22.set_ylabel("Mean width [km]")
    ax22.set_xlabel('A [km s-1]')
    ax22.tick_params(axis='both', which='major')
    ax22.tick_params(axis='both', which='minor')
    multi_fig.savefig('sharc_run/fig_for_paper/horizontal_slit_pscan_fixing.png')
    plt.close()

if quad_plot_cdata == True:
    path2_c_data = glob.glob('sharc_run/new_pscan/c_data/*/*')
    tilt_nb = [int(path2_c_data[i].split('_')[-3][1:]) for i in range(len(path2_c_data))]
    tilt_order_index = np.argsort(tilt_nb)
    path2_c_data = [path2_c_data[i] for i in tilt_order_index]
    max_L_cdata = []
    w_mean_data = []
    first_cdata = True
    for cdex, cdata_name in enumerate(path2_c_data):
        dumb_file = pd.read_csv(cdata_name)
        if dumb_file.shape[-1]<3:
            pass
        else:
            max_len_index = dumb_file['Max len [Mm]'].argmax()
            if first_cdata == True:
                first_cdata = False
                max_L_cdata = dumb_file.iloc[[max_len_index]]
                w_mean_cdata = pd.DataFrame([dumb_file.mean()])
            else:
                max_L_cdata = max_L_cdata.append(dumb_file.iloc[[max_len_index]],ignore_index=True)
                w_mean_cdata = w_mean_cdata.append(pd.DataFrame([dumb_file.mean()]),ignore_index=True)
    w_mean_cdata['Jet width [Mm]'] = w_mean_cdata['Jet width [Mm]']/km_to_Mm # covert jet widths to km despite name
    w_mean_cdata = w_mean_cdata.sort_values('Tilt angle [degree]',
                                            ascending=False)
    lw =  6
    multi_fig, ((ax11,ax12), (ax21, ax22)) = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(30,20))
    i = 0
    line_order = 0
    for key_t, grp_t in max_L_cdata.groupby(['Tilt angle [degree]'], sort=False):
        ground_nb = len(grp_t.groupby('Driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0        
        for key, grp in grp_t.groupby(['Magnetic field strength [B]']):
            for sub_key, sub_grp in grp.groupby(['Driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['Amplitude [km/s]'])
                j = shuff % len(styles)
                shuff += 1          
                plot_pd = sub_grp.plot(ax=ax21, kind='line', x='Amplitude [km/s]', 
                                  y='Max len [Mm]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j], lw=lw)
#                ax21.set_zorder(line_order) 
                line_order += 2
                if data_check == True:
                    print(sub_grp)
#        ax21.legend(loc=2)
#        ax21.set_ylabel("Max length [Mm]")
#        ax21.set_xlabel("A [km s-1]")
        if power_law_fit == True:
            mean_h = []
            std_h = []
            v = []
            for key, grp in max_L_cdata.groupby(['Amplitude [km/s]']):
                 mean_h.append(grp['Jet length [Mm]'].mean())
                 std_h.append(grp['Jet length [Mm]'].std())
                 v.append(key)
            if plot_error_bars == True:
                ax21.errorbar(v, mean_h, color='k', yerr=std_h, zorder=20, fmt='-o',
                             errorevery=1, barsabove=True, capsize=5, capthick=2)
            popt_log, pcov_log, ydatafit_log = curve_fit_log(v, mean_h, std_h)
            perr = np.sqrt(np.diag(pcov_log))
            nstd = 1 # to draw 5-sigma intervals
            popt_up = popt_log+nstd*perr
            popt_dw = popt_log-nstd*perr
            v = np.asarray(v)
    
            fit_up = np.power(10, linlaw(np.log10(v),*popt_up))
            fit_dw = np.power(10, linlaw(np.log10(v),*popt_dw))
            ax21.plot(v,ydatafit_log,'crimson',linewidth=lw, marker='o',
                     markersize=12, label='average')
            ax21.fill_between(v, fit_up, fit_dw, alpha=.25)
            test = powlaw(v, *popt_log) 
#    ax21.legend(fontsize=18)
    ax21.legend(loc=2)
    ax21.set_ylim(yliml21,ylimu21)
    ax21.set_ylabel("Max length [Mm]")
    ax21.set_xlabel('A [km s-1]')
    ax21.tick_params(axis='both', which='major')
    ax21.tick_params(axis='both', which='minor')
    i = 0
    line_order = 0
    for key_t, grp_t in max_L_cdata.groupby(['Tilt angle [degree]'], sort=False):
        ground_nb_t = len(grp_t.groupby('Driver time [s]').groups)
        ii = i*ground_nb_t % (len(colors)-1)
        i += 5
        shuff = 0
        for key, grp in grp_t.groupby(['Amplitude [km/s]']):
            for sub_key, sub_grp in grp.groupby(['Driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['Magnetic field strength [B]'])
                j = shuff % len(styles)
                shuff += 1
                plot_pd = sub_grp.plot(ax=ax11, kind='line',
                                  x='Magnetic field strength [B]', 
                                  y='Max len [Mm]',
                                  label='P='+str(int(sub_key))+', A='+str(int(key))+', '+r'$\theta=$'+str(int(key_t)),
                                  c=colors[ii], style=styles[j], lw=lw)
#                ax11.set_zorder(line_order) 
                line_order += 2
                if data_check == True:
                    print(sub_grp)
    ax11.set_xlim(xliml11,xlimu11)
    ax11.set_ylim(yliml11,ylimu11)
    ax11.legend(loc=1)
    ax11.set_ylabel("Max length [Mm]")
    ax11.set_xlabel('B [G]')
    ax11.tick_params(axis='both', which='major')
    ax11.tick_params(axis='both', which='minor')

    i = 0
    line_order = 0
    for key_s, grp_s in w_mean_cdata.groupby(['Tilt angle [degree]'], sort=False):
        ground_nb = len(grp_s.groupby('Driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0
        for key, grp in grp_s.groupby(['Amplitude [km/s]']):
            for sub_key, sub_grp in grp.groupby(['Driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['Magnetic field strength [B]'])
                j = shuff % len(styles)
                shuff += 1
                plot_pd = sub_grp.plot(ax=ax12, kind='line',
                                  x='Magnetic field strength [B]',
                                  y='Jet width [Mm]',
                                  label='P='+str(int(sub_key)) +
                                        ', A='+str(int(key)) + ', '+r'$\theta=$'+str(int(key_s)),
                                  c=colors[ii], style=styles[j], lw=lw)
#                ax12.set_zorder(line_order) 
                line_order += 2

                if data_check == True:
                    print(sub_grp)
    ax12.legend(loc=1)
#    handles, labels = ax12.get_legend_handles_labels()
#    ax12.legend(handles[::-1], labels[::-1], loc=1)
    ax12.set_ylabel("Mean width [km]")
    ax12.set_xlabel('B [G]')
    ax12.tick_params(axis='both', which='major')
    ax12.tick_params(axis='both', which='minor')

    i = 0
    line_order = 0
    for key_s, grp_s in w_mean_cdata.groupby(['Tilt angle [degree]'], sort=False):
        ground_nb = len(grp_s.groupby('Driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 5
        shuff = 0            
        for key, grp in grp_s.groupby(['Magnetic field strength [B]']):
            for sub_key, sub_grp in grp.groupby(['Driver time [s]']):
                sub_grp = sub_grp.sort_values(by=['Amplitude [km/s]'])
                j = shuff % len(styles)
                shuff += 1          
                plot_pd = sub_grp.plot(ax=ax22, kind='line', x='Amplitude [km/s]', 
                                  y='Jet width [Mm]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key))+', '+r'$\theta=$'+str(int(key_s)),
                                  c=colors[ii], style=styles[j], lw=lw)
#                ax22.set_zorder(line_order) 
                line_order += 2
                if data_check == True:
                    print(sub_grp)
    ax22.set_ylim(yliml22,ylimu22)
#    ax22.set_xlim(100,1800)
    ax22.legend(loc=2, ncol=2)
    ax22.set_ylabel("Mean width [km]")
    ax22.set_xlabel('A [km s-1]')
    ax22.tick_params(axis='both', which='major')
    ax22.tick_params(axis='both', which='minor')
    multi_fig.savefig('sharc_run/fig_for_paper/traced_slit_pscan_fixing.png')
    plt.close()

lhc_ax1.set_xlabel("Tilt ["+degree_sign+"]", fontsize=30)
lhc_ax1.set_ylabel("Max Height/Length [Mm]", fontsize=30)
lhc_ax2.set_ylabel("Max Height [Mm]", fontsize=30)
lhc_ax3.set_xlabel("Time [s]", fontsize=30)
lhc_ax3.set_ylabel("Length [Mm]", fontsize=30)
lhc_ax3.set_xlim(0,1100)

lhc_ax1.tick_params(axis='both', which='major', labelsize=28)
lhc_ax2.tick_params(axis='both', which='major', labelsize=28)
lhc_ax3.tick_params(axis='both', which='major', labelsize=28)

lhc_ax1.legend(prop={'size': 24})
lhc_ax2.legend(ncol=2, prop={'size': 18})
lhc_ax3.legend(ncol=2, prop={'size': 18})
plt.tight_layout()
fig_len_h_comp.savefig('sharc_run/fig_for_paper/combine_L_h_comp.png', bbox_inches='tight')
plt.close()