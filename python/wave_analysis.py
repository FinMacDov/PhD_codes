import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg') # revert above
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from scipy.interpolate import UnivariateSpline
import pickle
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly
import scaleogram as scg 
import pywt
from pathlib import Path

# for CT_wavelet
from waveletFunctions import wavelet, wave_signif
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
i = 0
shuff = 0
SMALL_SIZE = 30
MEDIUM_SIZE = SMALL_SIZE + 2
BIGGER_SIZE = MEDIUM_SIZE + 2

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## old data
#path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'    
#dir_paths =  glob.glob('data/*')

## new data
path_2_shared_drive = '/run/user/1000/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'
data_save_dir = 'hdt/detrended_data'     


## Newer data
#path_2_shared_drive = '/run/user/1000/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j'
#data_save_dir = 'hdt/detrended_data_alt'     

# orig data run 
dir_paths =  glob.glob('data/high_dt/*')

## new data run
#dir_paths =  glob.glob('hdt/*')

## even newer data
#driver_time = '300'
#mag_str = '50'
#amplitude = '50'
#dir_paths = glob.glob('hdt_alt/jet_P' + driver_time +
#                      '_B' + mag_str +
#                      'A_'+amplitude + '_/*')

## newer data
#rise_window = [15]
#rise_poly_n = [2]
#
#fall_window = [20]
#fall_poly_n = [3]

# newer data
rise_window = [15]
rise_poly_n = [2]

fall_window = [10]
fall_poly_n = [2]

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

# I messed up time scaling on data collection
#TIME_CORRECTION_FACTOR = 10/unit_time

#orig run
TIME_CORRECTION_FACTOR = 1e-1

#TIME_CORRECTION_FACTOR = 1
#dt = unit_time/20
dt = unit_time/200 # high dt

unit_mass = unit_density*unit_length**3
unit_specific_energy = (unit_length/unit_time)**2

# options
plot_w_vs_t = True
testing = True
CT_wavelet_code = False
save_pd_cvs = False
HoI = 2 #  Mm
all_data = False # plots all data as suppose to small selection 
sf = [0.40, 0.35, 0.3, 0.3]
plot_mean_w_vs_BAdt = False

lw =  3# 2.5#  

## orig
#rise_window = [15]
#fall_window = [10]

##new data
## reviesd
#rise_window = [5, 20, 15,  15,  10,  15]
#rise_poly_n = [2,  2,  3,   4,   2,   3]
#
#fall_window = [10, 15, 20, 15,   20,  20]
#fall_poly_n = [2,   3,  2, 2,    3,   3]


# how to read pickels
max_h_data_set = pd.read_pickle(dir_paths[1])
big_data_set = pd.read_pickle(dir_paths[0])

#max_h_data_set = pd.read_pickle(dir_paths[-2])
#big_data_set = pd.read_pickle(dir_paths[-1])

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
list_of_indexs = [[20],[40],[60],[80]]    

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
        data_save_dir_2 = data_save_dir+'/P'+str(big_data_set['idx'][idx][0]) + \
                        '/B'+str(big_data_set['idx'][idx][1])+ \
                        '/A'+str(big_data_set['idx'][idx][2])
        if not os.path.exists(w_path):
            os.makedirs(w_path)
        Path(data_save_dir_2).mkdir(parents=True, exist_ok=True)
        lab = 'P=' + str(big_data_set['idx'][idx][0]) + ' s' + \
        ', B=' + str(big_data_set['idx'][idx][1]) + ' G' + \
        ', A=' + str(big_data_set['idx'][idx][2]) + r' $\rm{km~s^{-1}}$'
        data_x, data_y = clipped_h_data_plotter(big_data_set, idx)
#        data_y = []   
        data_x = data_x*dt

        jet_width = np.asanyarray(big_data_set['dfs'][idx]['jet Width [km]'])
        jet_time = np.asanyarray(big_data_set['dfs'][idx]['side time [s]'])

        jet_time = jet_time*TIME_CORRECTION_FACTOR
        height_markers = np.asanyarray(big_data_set['dfs'][idx]['height [Mm]'])

        if np.isnan(sum(big_data_set['dfs'][idx]['height [Mm]']))==True:
             pass
        else:
            data_mean_w.append([big_data_set['idx'][idx][0],
                                big_data_set['idx'][idx][1],
                                big_data_set['idx'][idx][2],
                                big_data_set['dfs'][idx]['jet Width [km]'].mean()])
            h_index = np.argwhere(height_markers==HoI)
            i += 1
            i = i % len(colors)
            j = i % len(styles_alt)
            selected_jet_width = jet_width[h_index]
            selected_time = jet_time[h_index]
            selected_jet_width = selected_jet_width.reshape(len(selected_jet_width))
            selected_time = selected_time.reshape(len(selected_jet_width))
            if save_pd_cvs==True:
#                Path(data_save_dir_2).mkdir(parents=True, exist_ok=True)   
                orig_data = pd.DataFrame(data={'time': selected_time,'jet width [km]': selected_jet_width})
                orig_data.to_csv(data_save_dir_2+'/original_data_data.csv', index = True, header=True)

            detrended_jet_width = []
            detrended_jet_width_2 = signal.detrend(selected_jet_width)

            for ind in range(1,len(selected_jet_width)):
                detrended_jet_width.append(selected_jet_width[ind]-selected_jet_width[ind-1])   

#            detrended_jet_width = selected_jet_width-selected_jet_width.mean()
            # get riase abnd fall
            max_h_idx = big_data_set['dfs'][idx]['Height [Mm]'].dropna().argmax()
#            max_h_time = big_data_set['dfs'][0]['time [s]'].dropna()[max_h_idx]*TIME_CORRECTION_FACTOR*dt
            max_h_time = big_data_set['dfs'][idx]['time [s]'].dropna()[max_h_idx]*TIME_CORRECTION_FACTOR

            jet_fall_indexs = np.argwhere(selected_time >= max_h_time)
            jet_rise_indexs = np.argwhere(selected_time <= max_h_time)

            rise_time = selected_time[jet_rise_indexs].reshape(len(jet_rise_indexs))
            rise_widths = selected_jet_width[jet_rise_indexs].reshape(len(jet_rise_indexs))
            coefs_rise = poly.polyfit(rise_time,rise_widths,rise_poly_n[idx])
            ffit_rise = poly.polyval(rise_time, coefs_rise)
            poly_detrend_rise = rise_widths-ffit_rise

            fall_time = selected_time[jet_fall_indexs].reshape(len(jet_fall_indexs))
            fall_widths = selected_jet_width[jet_fall_indexs].reshape(len(jet_fall_indexs))
            coefs_fall = poly.polyfit(fall_time,fall_widths,fall_poly_n[idx])
            ffit_fall = poly.polyval(fall_time, coefs_fall)
            poly_detrend_fall = fall_widths-ffit_fall
            # orig data
            ii = 0
            if testing == True:
                fig, ax = plt.subplots(figsize=(20,12))
                for hi in range(1,int(max(height_markers))):
                    ii += 1
                    ii = ii % len(colors)
                    jj = ii % len(styles_alt)
                    if ii ==0: styles_alt = [styles_alt[-1]]+styles_alt[:-1]
                    h_index = np.argwhere(height_markers==hi)
                    ax.plot(jet_time[h_index], jet_width[h_index], color=colors[ii], linestyle=styles_alt[jj],
                            lw=lw+3, label='height='+str(hi)+' Mm')
                ax.legend(ncol=2)
                ax.set_title(lab)
                ax.set_ylabel("Jet width [km]")
                ax.set_xlabel("Time [s]")
                ax.tick_params(axis='both', which='major')
                ax.tick_params(axis='both', which='minor')
    #            plt.savefig(w_path+'/P'+str(big_data_set['idx'][idx][0])+ \
    #                '_B'+str(big_data_set['idx'][idx][1])+ \
    #                '_A'+str(big_data_set['idx'][idx][2])+'.png')
    #        plt.close()
                fig.savefig(data_save_dir_2+'/P'+str(big_data_set['idx'][idx][0])+
                        'B'+str(big_data_set['idx'][idx][1])+'A'+str(big_data_set['idx'][idx][2])+'.png')#,dpi=600)
                fig.clf()
                plt.close()
            # trajectory plots
            if testing == True:
                fig, ax = plt.subplots(figsize=(20,12))
                ax.set_title(lab, fontsize=25)
                ax.plot(data_x,data_y)
                ax.axvline(max_h_time,color = 'k')
                ax.set_ylabel("jet height [Mm]", fontsize=25)
                ax.set_xlabel("time [s]", fontsize=25)
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.tick_params(axis='both', which='minor', labelsize=18)
    #            plt.savefig(w_path+'/P'+str(big_data_set['idx'][idx][0])+ \
    #                '_B'+str(big_data_set['idx'][idx][1])+ \
    #                '_A'+str(big_data_set['idx'][idx][2])+'.png')
    #        plt.close()
                fig.savefig(data_save_dir_2+'/P'+str(big_data_set['idx'][idx][0])+
                        'B'+str(big_data_set['idx'][idx][1])+'A'+str(big_data_set['idx'][idx][2])+'_trajectory.png')#,dpi=600)
                fig.clf()
                plt.close()
            # spiltting data in 2
            if testing == True:
                fig, ax = plt.subplots(figsize=(20,12))
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
                rise_line, = ax.plot(rise_time, rise_widths,
                        color=colors[i+2], linestyle=styles_alt[0],
                        lw=lw, marker='o', label='rise')
                fall_line, = ax.plot(fall_time, fall_widths, color=colors[i+1], 
                        linestyle=styles_alt[0], lw=lw, 
                        marker='o', label = 'fall')
                rise_dtrend, = ax.plot(rise_time, ffit_rise, label='rise detrend')
                fall_dtrend, = ax.plot(fall_time, ffit_fall, label='fall detrend')
                # apex of jet
                ax.axvline(max_h_time,color = 'k')
#                plt.title('raw data with detrend lines')
                ax.set_ylabel("Jet width [km]")
                ax.set_xlabel("Time [s]")
                plt.legend(handles=[rise_line, fall_line, rise_dtrend, fall_dtrend])
                fig.savefig(data_save_dir_2+'/raw data_P'+str(big_data_set['idx'][idx][0])+
                            'B'+str(big_data_set['idx'][idx][1])+'A'+str(big_data_set['idx'][idx][2])+'.png')#,dpi=600)
                fig.clf()
                plt.close()

            # detrended polly fits
            #add rolling average to data
            rise_panda = pd.DataFrame(data={'time': rise_time,'detrend': poly_detrend_rise})
            fall_panda = pd.DataFrame(data={'time': fall_time,'detrend': poly_detrend_fall})
            rolling_rise_mean = rise_panda.detrend.rolling(window=rise_window[idx]).mean()
            rolling_fall_mean = fall_panda.detrend.rolling(window=fall_window[idx]).mean()
            rolling_rise_mean = pd.concat([rise_panda.time, rolling_rise_mean], axis=1) 
            rolling_fall_mean = pd.concat([fall_panda.time, rolling_fall_mean], axis=1) 
            
            if save_pd_cvs==True:
                rise_panda.to_csv(data_save_dir_2+'/detrended_rise_data.csv', index = True, header=True)
                fall_panda.to_csv(data_save_dir_2+'/detrended_fall_data.csv', index = True, header=True)

                rolling_rise_mean.to_csv(data_save_dir_2+'/rise_data.csv', index = True, header=True)
                rolling_fall_mean.to_csv(data_save_dir_2+'/fall_data.csv', index = True, header=True) 
                
                info_file = open(data_save_dir_2+'/info.txt', 'w')
                info_file.write('rise_window = ' + str(rise_window)+ '\n' +
                               'rise_poly_n = ' + str(rise_poly_n) + '\n'+ '\n' +
                               'fall_window = ' + str(fall_window) + '\n' +
                               'fall_poly_n = ' + str(fall_poly_n))
                info_file.close() 
            if testing == True:
                fig, ax = plt.subplots(figsize=(20,12))
                rise_dtrend, = plt.plot(rise_time, poly_detrend_rise,
                                      marker='o', label='rise detrend')
                rise_smoothed, = plt.plot(rise_panda.time, rolling_rise_mean.detrend, 'r',
                         label='rise smoothed')
    
                fall_dtrend, = plt.plot(fall_time, poly_detrend_fall,
                                        marker='o', label='fall detrend')
                fall_smoothed, = plt.plot(fall_panda.time,
                                          rolling_fall_mean.detrend, 
                                          color='purple', label='fall smoothed')
                plt.legend(handles=[rise_dtrend, rise_smoothed, fall_dtrend, fall_smoothed])
                plt.title('detrended data with smoothed data')
                fig.savefig(data_save_dir_2+'/detrended_P'+str(big_data_set['idx'][idx][0])+
                        'B'+str(big_data_set['idx'][idx][1])+'A'+str(big_data_set['idx'][idx][2])+'.png')#,dpi=600)
                fig.clf()
                plt.close()
            
            
#            # fft python on raw data
#            sp = np.fft.fft(selected_jet_width)
#            freq = np.fft.fftfreq(selected_time.shape[-1])
#            fig, ax = plt.subplots(figsize=(20,12))
#            plt.plot(freq, sp.real, freq, sp.imag)
#            plt.show()

            # fft python on detrnded data
            sp = np.fft.fft(poly_detrend_fall)
            freq = np.fft.fftfreq(fall_time.shape[-1])
#            if testing == True:
#                fig, ax = plt.subplots(1, 1, figsize=(20,12))
#                plt.plot(freq, sp.real, freq, sp.imag)
#                plt.xlabel('frequency (Hz)')
#                plt.ylabel('Amplitude (km)')
#                plt.xlim(-0.1,0.1)
#                plt.show()

            sp = np.fft.fft(poly_detrend_rise)
            freq = np.fft.fftfreq(rise_time.shape[-1])
#            if testing == True:
#                fig, ax = plt.subplots(figsize=(20,12))
#                plt.plot(freq, sp.real, freq, sp.imag)
#                plt.xlabel('frequency (Hz)')
#                plt.ylabel('Amplitude (km)')
#                plt.xlim(-0.1,0.1)
#                plt.show()


            # Wavelet attempt
            wl_rise_data = rolling_rise_mean.dropna()
            wl_rise_time = rise_panda.time[wl_rise_data.index].to_numpy()
            wl_rise_data = wl_rise_data.to_numpy()
            wl_rise_data_norm = wl_rise_data-wl_rise_data.mean()
            # choose default wavelet function for the entire notebook
#            scg.set_default_wavelet('cmor1.5-1.0')
#            if testing == True:
#                fig1, axs = plt.subplots(2, figsize=(20,12)); 
#                lines = axs[0].plot(wl_rise_time, wl_rise_data);
#                axs[0].set(xlabel='time',
#                           ylabel='detrended and norm width of jet [km]')
#                           
#                scales = scg.periods2scales(np.arange(1, 120))
#                scg.cws(wl_rise_time, wl_rise_data_norm, scales=scales, ax=axs[1])
#    
#                scg.cws(wl_rise_time, wl_rise_data_norm, ax=axs[1],
#                        yscale='linear', cbar='horizontal'); 
#                plt.tight_layout()
#                plt.show()
    
            wl_fall_data = rolling_fall_mean.dropna()
            wl_fall_time = fall_panda.time[wl_fall_data.index].to_numpy()
            wl_fall_time = wl_fall_time-wl_fall_time[0]
            wl_fall_data = wl_fall_data.to_numpy()
            wl_fall_data_norm = wl_fall_data-wl_fall_data.mean()
            # choose default wavelet function forlen() the entire notebook
#            if testing == True:
#                fig1, axs = plt.subplots(2, figsize=(20,12)); 
#                lines = axs[0].plot(wl_fall_time, wl_fall_data);
#                axs[0].set(xlabel='time',
#                           ylabel='detrended and norm width of jet [km]')
#                axs[0].set_xlim(0,max(wl_fall_time))
#    
#                scales = scg.periods2scales(np.arange(1, 60)) # 20
#                scg.cws(wl_fall_time, wl_fall_data_norm, scales=scales, ax=axs[1], 
#                        cbar='horizontal'); 
#    
#                plt.tight_layout()
#                plt.show()

            if CT_wavelet_code==True:
                # READ THE DATA
                # orig data
#                sst = np.loadtxt('sea_data/sst_nino3.txt')  # input SST time series
                sst = wl_fall_data  # input SST time series
                sst = sst - np.mean(sst)
                variance = np.std(sst, ddof=1) ** 2
                print("variance = ", variance)
                
                #----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E------------------------------------------------------
                
                # normalize by standard deviation (not necessary, but makes it easier
                # to compare with plot on Interactive Wavelet page, at
                # "http://paos.colorado.edu/research/wavelets/plot/"
                if 0:
                    variance = 1.0
                    sst = sst / np.std(sst, ddof=1)
                n = len(sst)
                dt = 0.01
#                time = np.arange(len(sst)) * dt + 1871.0  # construct time array
                time =  wl_fall_time # construct time array
                xlim = ([min(time), max(time)])  # plotting range
                pad = 1  # pad the time series with zeroes (recommended)
                dj = 0.25  # this will do 4 sub-octaves per octave
                s0 = 2 * dt  # this says start at a scale of 6 months
                j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
                lag1 = 0.72  # lag-1 autocorrelation for red noise background
                print("lag1 = ", lag1)
                mother = 'MORLET'
                
                # Wavelet transform:
                wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
                power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
                global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
                
                # Significance levels:
                signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                    lag1=lag1, mother=mother)
                sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
                sig95 = power / sig95  # where ratio > 1, power is significant
                
                # Global wavelet spectrum & significance levels:
                dof = n - scale  # the -scale corrects for padding at edges
                global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
                    lag1=lag1, dof=dof, mother=mother)
                
                # Scale-average between El Nino periods of 2--8 years
                avg = np.logical_and(scale >= 2, scale < 8)
                Cdelta = 0.776  # this is for the MORLET wavelet
                scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
                scale_avg = power / scale_avg  # [Eqn(24)]
                scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
                scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
                    lag1=lag1, dof=([2, 7.9]), mother=mother)
                
                #------------------------------------------------------ Plotting
                
                #--- Plot time series
                fig = plt.figure(figsize=(9, 10))
                gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
                plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
                plt.subplot(gs[0, 0:3])
                plt.plot(time, sst, 'k')
                plt.xlim(xlim[:])
                plt.xlabel('Time (s)')
                plt.ylabel('Detrended jet width (km)')
                
                plt.text(time[-1] + 35, 0.5,'Wavelet Analysis\nC. Torrence & G.P. Compo\n' +
                    'http://paos.colorado.edu/\nresearch/wavelets/',
                    horizontalalignment='center', verticalalignment='center')
                
                #--- Contour plot wavelet power spectrum
                # plt3 = plt.subplot(3, 1, 2)
                plt3 = plt.subplot(gs[1, 0:3])
                levels = [0, 0.5, 1, 2, 4, 999]
                CS = plt.contourf(time, period, power, len(levels))  #*** or use 'contour'
                im = plt.contourf(CS, levels=levels, colors=['white','bisque','orange','orangered','darkred'])
                plt.xlabel('Time (s)')
                plt.ylabel('Period (s)')
                plt.xlim(xlim[:])
                # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
                plt.contour(time, period, sig95, [-99, 1], colors='k')
                # cone-of-influence, anything "below" is dubious
                plt.plot(time, coi, 'k')
                # format y-scale
                plt3.set_yscale('log', basey=2, subsy=None)
                plt.ylim([np.min(period), np.max(period)])
                ax = plt.gca().yaxis
                ax.set_major_formatter(ticker.ScalarFormatter())
                plt3.ticklabel_format(axis='y', style='plain')
                plt3.invert_yaxis()
                # set up the size and location of the colorbar
                # position=fig.add_axes([0.5,0.36,0.2,0.01]) 
                # plt.colorbar(im, cax=position, orientation='horizontal') #, fraction=0.05, pad=0.5)
                
                # plt.subplots_adjust(right=0.7, top=0.9)
                
                #--- Plot global wavelet spectrum
                plt4 = plt.subplot(gs[1, -1])
                plt.plot(global_ws, period)
                plt.plot(global_signif, period, '--')
                plt.xlabel('Power')
                plt.title('c) Global Wavelet Spectrum')
                plt.xlim([0, 1.25 * np.max(global_ws)])
                # format y-scale
                plt4.set_yscale('log', basey=2, subsy=None)
                plt.ylim([np.min(period), np.max(period)])
                ax = plt.gca().yaxis
                ax.set_major_formatter(ticker.ScalarFormatter())
                plt4.ticklabel_format(axis='y', style='plain')
                plt4.invert_yaxis()
                
                # --- Plot 2--8 yr scale-average time series
                plt.subplot(gs[2, 0:3])
                plt.plot(time, scale_avg, 'k')
                plt.xlim(xlim[:])
                plt.xlabel('Time (s)')
                plt.ylabel('Avg variance')
                plt.plot(xlim, scaleavg_signif + [0, 0], '--')
                
                plt.show()
            
            
            
            
#            deg = 3
##            t_new = np.linspace(selected_time[0], selected_time[-1],500)            
#            coefs = poly.polyfit(selected_time,selected_jet_width,deg)
#            ffit = poly.polyval(selected_time, coefs)
            
#            # plot of polynomial
#            ax.plot(selected_time, ffit)
#            ax.plot(jet_time[h_index], jet_width[h_index], color=colors[i+1], linestyle=styles_alt[0],
#                    lw=lw, label='height='+str(HoI)+' Mm', marker='o')
#           #detredned data
#            fig, ax = plt.subplots(figsize=(20,12))
#            poly_detrend = selected_jet_width-ffit
#            ax.plot(selected_time, poly_detrend)
#            plt.show()

#            # plot of difs
#            ax.plot(detrended_jet_width, color=colors[i], linestyle=styles_alt[0],
#                    lw=lw+3, label='height='+str(HoI)+' Mm')
            # using signal thing (I think it for linear trends)
#            ax.plot(detrended_jet_width_2, color=colors[i+1], linestyle=styles_alt[0],
#                    lw=lw+3, label='height='+str(HoI)+' Mm')
#            plt.show()


#            # linear regression
#            fig, ax = plt.subplots(figsize=(20,12))
#            model = LinearRegression()
#            model.fit(selected_time,selected_jet_width)
#            trend = model.predict(selected_time)
#            detrended = [selected_jet_width[i]-trend[i] for i in range(0, len(selected_jet_width))]
#            plt.plot(detrended)
#            plt.show()

