import sys
import random as rd
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg') # revert above
import matplotlib.pyplot as plt
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

i = 0
shuff = 0
SMALL_SIZE = 40
MEDIUM_SIZE = SMALL_SIZE+2
BIGGER_SIZE = MEDIUM_SIZE+2

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=22)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
path_2_shared_drive = '/run/user/1000/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
#dir_paths =  glob.glob('data/*')
#data set for paper
dir_paths =  glob.glob('data/run3/*')
##dat srt for high dt
#dir_paths =  glob.glob('data/high_dt/*')
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
all_data = False # plotss all data as suppose to small selection 
plot_error_bars = False
power_law_fit = False
plot_hmax_vs_dt = False
data_check = False
interp_check = False # Doesnt work well enough for my purposes
diff_check = False
sf = [0.65, 0.62, 0.62, 0.62]
plot_mean_w_vs_BAdt = False
test_balstic = False
Decelleration_analysis = False


lw =  3# 2.5#  

# how to read pickels
max_h_data_set = pd.read_pickle(dir_paths[1])
big_data_set = pd.read_pickle(dir_paths[0])

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
styles = ['-', '--', '-.', ':','-', '--', '-.', ':','-']
styles_alt = ['-', '--', '-.', ':']

plt.rc('lines', linewidth=lw)


multi_fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(30,20))

#list_of_indexs = [[300,20,80],[200,20,80],[50,20,80],[300,80,80],[300,100,80]]

list_of_indexs = [[20],[40],[60],[80]]

#list_of_indexs = [[60]]

spear_list = []
decell_array = []
vmax_array = []
predicted_decell_array = []

nr = 0
nc = 0 
if plot_h_vs_t == True:
    i=0
    if all_data == True:
        fig, ax = plt.subplots()
        for idx in range(len(big_data_set)):
            i = idx % len(colors)
            j = idx % len(styles_alt)
            big_data_plotter(big_data_set, x_name='time [s]', y_name='Height [Mm]',
                             index=idx, ax=ax, label='a',
                             colour=colors[i], style=styles_alt[j], lw=lw,
                             figsize=(20,12))
        plt.show()
    else:
        for idx_num, idx_name in enumerate(list_of_indexs):
            #ax = plt.gca()
            if nc==1:
                nr += 1
            nc = idx_num % 2
            idx_loc = [iloc for iloc, eye in enumerate(big_data_set['idx']) if sum(abs(eye[-1]-idx_name))==0]
#            idx_loc = [50] # fudge to make code run single case
            if data_check == True:
                for iloc, eye in enumerate(big_data_set['idx']):
                    print(iloc,eye[0],idx_name,eye-idx_name,
                          sum(eye-idx_name),sum(abs(eye-idx_name))==0)
            t_max = 0
            for idx in idx_loc:
                i += 1
                i = i % len(colors)
                j = i % len(styles_alt)
                if i ==0: styles_alt = [styles_alt[-1]]+styles_alt[:-1]
                lab = 'P=' + str(big_data_set['idx'][idx][0])+ \
                ', B='+str(big_data_set['idx'][idx][1])+ \
                ', A='+str(big_data_set['idx'][idx][2])
                      
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
                ax[nr,nc].plot(data_x, data_y, color=colors[i], linestyle=styles_alt[j],
                         lw=lw+3, label=lab)
            #for ballstic test, goes here
            ax[nr,nc].set_xlim(right=t_max+t_max*sf[idx_num])
            ax[nr,nc].legend(loc=1)
            ax[nr,nc].set_ylabel("Maximum height [Mm]")
            ax[nr,nc].set_xlabel("t [s]")
            ax[nr,nc].tick_params(axis='both', which='major')
            ax[nr,nc].tick_params(axis='both', which='minor')
#            plt.savefig('P'+str(big_data_set['idx'][idx][0])+ \
#                        '_B'+str(big_data_set['idx'][idx][1])+ \
#                        '_A'+str(big_data_set['idx'][idx][2])+'_BF.png')
#            plt.close()
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
            plt.plot(predicted_decel_p50,vmax_scan, label='P=50s')
            plt.plot(predicted_decel_p200,vmax_scan, label='P=200s')
            plt.plot(predicted_decel_p300,vmax_scan, label='P=300s')
            test_ax.legend(loc=4, fontsize=24)
            test_ax.set_ylabel("maximum V [km s-1]", fontsize=32)
            test_ax.set_xlabel("decelerations [m s-2]", fontsize=32)
            test_ax.tick_params(axis='both', which='major', labelsize=28)
            test_ax.tick_params(axis='both', which='minor', labelsize=28)
            plt.show()
        # doesnt work for pandas
plt.savefig('sj_paper/jet_traj.png')
plt.close()
