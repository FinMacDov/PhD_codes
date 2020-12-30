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
SMALL_SIZE = 24
MEDIUM_SIZE = SMALL_SIZE+2
BIGGER_SIZE = MEDIUM_SIZE+2

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-11)    # legend fontsize
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
plot_h_vs_t = False
plot_w_vs_t = True
plot_error_bars = False
plot_hmax_vs_B = True
plot_hmax_vs_A = True
power_law_fit = True
plot_hmax_vs_dt = True
data_check = False
sf = [0.60, 0.55, 0.5, 0.5]
plot_mean_w_vs_BAdt = True

lw =  3# 2.5#  

# how to read pickels
max_h_data_set = pd.read_pickle(dir_paths[1])
big_data_set = pd.read_pickle(dir_paths[0])


colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
styles = ['-', '--', '-.', ':','-', '--', '-.', ':','-']
styles_alt = ['-', '--', '-.', ':']

plt.rc('lines', linewidth=lw)

list_of_indexs = [[60]]

multi_fig, ((ax11,ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(30,20))

spear_list = []
decell_array = []
vmax_array = []
predicted_decell_array = []

data_mean_w = []
if plot_w_vs_t == True:
    i=0
    t_max = 0
    w_root_dir = 'width_graphs'
    if not os.path.exists(w_root_dir):
        os.makedirs(w_root_dir)
    for idx in range(len(big_data_set)):
        w_path = w_root_dir+'/P'+str(big_data_set['idx'][idx][0]) + \
                        '/B'+str(big_data_set['idx'][idx][1])
        if not os.path.exists(w_path):
            os.makedirs(w_path)
        lab = 'P=' + str(big_data_set['idx'][idx][0])+ \
        ', B='+str(big_data_set['idx'][idx][1])+ \
        ', A='+str(big_data_set['idx'][idx][2])
        data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
        data_y = []        
        t_max = max(data_x)*TIME_CORRECTION_FACTOR*dt
        jet_width = np.asanyarray(big_data_set['dfs'][idx]['jet Width [km]'])
        jet_time = np.asanyarray(big_data_set['dfs'][idx]['side time [s]'])*TIME_CORRECTION_FACTOR*dt
        height_markers = np.asanyarray(big_data_set['dfs'][idx]['height [Mm]'])
        if np.isnan(sum(big_data_set['dfs'][idx]['height [Mm]']))==True:
             pass
        else:
            data_mean_w.append([big_data_set['idx'][idx][0],
                                big_data_set['idx'][idx][1],
                                big_data_set['idx'][idx][2],
                                big_data_set['dfs'][idx]['jet Width [km]'].mean()])
            for hi in range(1,int(max(height_markers))):
                i += 1
                i = i % len(colors)
                j = i % len(styles_alt)
                if i ==0: styles_alt = [styles_alt[-1]]+styles_alt[:-1]
                h_index = np.argwhere(height_markers==hi)

    df_w = pd.DataFrame(data_mean_w, columns=['driver time [s]',
                                              'magnetic field strength [G]',
                                              'amplitude [km s-1]',
                                              'mean width [km]'])
    if plot_mean_w_vs_BAdt == True:
        wmean_root_dir = w_root_dir+'/width_mean_graphs'
        if not os.path.exists(wmean_root_dir):
            os.makedirs(wmean_root_dir)
        i=0
#        fig, ax = plt.subplots(figsize=(20,12))
        for key, grp in df_w.groupby(['amplitude [km s-1]']):
            ground_nb = len(grp.groupby('driver time [s]').groups)
            ii = i*ground_nb % (len(colors)-1)
            i += 1
            shuff = 0
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                j = shuff % len(styles)
                shuff += 1
                pd_plot = sub_grp.plot(ax=ax12, kind='line',
                                  x='magnetic field strength [G]', 
                                  y='mean width [km]',
                                  label='P='+str(int(sub_key))+', A='+str(int(key)),
                                  c=colors[ii], style=styles[j])
                if data_check == True:
                    print(sub_grp)
        ax12.legend(loc=1)
        ax12.set_ylabel("Mean width [km]")
        ax12.set_xlabel('B [G]')
        ax12.tick_params(axis='both', which='major')
        ax12.tick_params(axis='both', which='minor')

        i = 0
        for key, grp in df_w.groupby(['magnetic field strength [G]']):
            ground_nb = len(grp.groupby('driver time [s]').groups)
            ii = i*ground_nb % (len(colors)-1)
            i += 1
            shuff = 0
            for sub_key, sub_grp in grp.groupby(['driver time [s]']):
                j = shuff % len(styles)
                shuff += 1          
                pd_plot = sub_grp.plot(ax=ax32, kind='line', x='amplitude [km s-1]', 
                                  y='mean width [km]',
                                  label='P='+str(int(sub_key))+', B='+str(int(key)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
        ax32.legend(loc=1)
        ax32.set_ylabel("Mean width [km]")
        ax32.set_xlabel('A [km s-1]')
        ax32.tick_params(axis='both', which='major')
        ax32.tick_params(axis='both', which='minor')
        ax32.set_xlim(right=88.5)
        for key, grp in df_w.groupby(['magnetic field strength [G]']):
            ground_nb = len(grp.groupby('driver time [s]').groups)
            ii = i*ground_nb % (len(colors)-1)
            i += 1
            shuff = 0 
            for sub_key, sub_grp in grp.groupby(['amplitude [km s-1]']):
                j = shuff % len(styles)
                shuff += 1
                plt_pd = sub_grp.plot(ax=ax22, kind='line', x='driver time [s]', 
                                  y='mean width [km]',
                                  label='B='+str(int(key))+', A='+str(int(sub_key)),
                                  c=colors[ii], style=styles[j], lw=lw)
                if data_check == True:
                    print(sub_grp)
        ax22.legend(loc=1)
        ax22.set_ylabel("Mean width [km]")
        ax22.set_xlabel('P [s]')
        ax22.tick_params(axis='both', which='major')
        ax22.tick_params(axis='both', which='minor')
        ax22.set_xlim(right=350)



if plot_hmax_vs_B == True:
    i=0
    for key, grp in max_h_data_set.groupby(['amplitude [km s-1]']):
        ground_nb = len(grp.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 1
        shuff = 0
        for sub_key, sub_grp in grp.groupby(['driver time [s]']):
            j = shuff % len(styles)
            shuff += 1
            pd_plot = sub_grp.plot(ax=ax11, kind='line',
                              x='magnetic field strength [G]', 
                              y='max height [Mm]',
                              label='P='+str(int(sub_key))+', A='+str(int(key)),
                              c=colors[ii], style=styles[j])
            if data_check == True:
                print(sub_grp)
    ax11.set_xlim(right=115)
    ax11.legend(loc=1)
    ax11.set_ylabel("Max height [Mm]")
    ax11.set_xlabel('B [G]')
    ax11.tick_params(axis='both', which='major')
    ax11.tick_params(axis='both', which='minor')

if plot_hmax_vs_A == True:
    i = 0
    for key, grp in max_h_data_set.groupby(['magnetic field strength [G]']):
        ground_nb = len(grp.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 1
        shuff = 0
        for sub_key, sub_grp in grp.groupby(['driver time [s]']):
            j = shuff % len(styles)
            shuff += 1          
            pd_plot = sub_grp.plot(ax=ax31, kind='line', x='amplitude [km s-1]', 
                              y='max height [Mm]',
                              label='P='+str(int(sub_key))+', B='+str(int(key)),
                              c=colors[ii], style=styles[j], lw=lw)
            if data_check == True:
                print(sub_grp)
    ax31.legend(loc=2)
    ax31.set_ylabel("Max height [Mm]")
    ax31.set_xlabel("A [km s-1]")
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
        ax31.plot(v,ydatafit_log,'crimson',linewidth=lw, marker='o',
                 markersize=12, label='average')
        ax31.fill_between(v, fit_up, fit_dw, alpha=.25)
        ax31.set_ylabel("Max height [Mm]")
        ax31.set_xlabel('A [km s-1]')
        ax31.tick_params(axis='both', which='major')
        ax31.tick_params(axis='both', which='minor')
        test = powlaw(v, *popt_log)

if plot_hmax_vs_dt == True:
    for key, grp in max_h_data_set.groupby(['magnetic field strength [G]']):
        ground_nb = len(grp.groupby('driver time [s]').groups)
        ii = i*ground_nb % (len(colors)-1)
        i += 1
        shuff = 0 
        for sub_key, sub_grp in grp.groupby(['amplitude [km s-1]']):
            j = shuff % len(styles)
            shuff += 1
            pd_plot = sub_grp.plot(ax=ax21, kind='line', x='driver time [s]', 
                              y='max height [Mm]',
                              label='B='+str(int(key))+', A='+str(int(sub_key)),
                              c=colors[ii], style=styles[j], lw=lw, loglog=False)
            if data_check == True:
                print(sub_grp)

    ax21.set_xlim(right=350)
    ax21.legend(loc=1)
    ax21.set_ylabel("Max height [Mm]")
    ax21.set_xlabel('P [s]')
    ax21.tick_params(axis='both', which='major')
    ax21.tick_params(axis='both', which='minor')

#plt.tight_layout()
plt.savefig('test_combine_image.png')
plt.close()
