# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:36:22 2018

@author: fionnlagh
"""
import os
from collections import OrderedDict

# create parfile info
filelist = OrderedDict([('base_filename=', "'test'"),
                        ('saveprim=', '.T.'),
                        ('autoconvert=', '.T.'),
                        ('convert_type=', "'vtuBCCmpi'"),
                        ('nwauxio=', '9')])

savelist = OrderedDict([('dtsave_log=', '0.1'),
                        ('itsave(1,1)=', '0'),
                        ('itsave(1,2)=', '0'),
                        ('dtsave=', '0.01d0,0.05d0')])

stoplist = OrderedDict([('dtmin=', '1.0d-12'),
                        ('time_max=', '10.0d0')])

methodlist = OrderedDict([('time_integrator=', "'threestep'"),
                          ('flux_scheme=', '20*'+"'hll'"),
                          ('limiter=', '20*'+"'cada3'"),
                          ('small_values_method=', "'average'"),
                          ('small_values_daverage=', '4')])

boundlist = OrderedDict([('nghostcells=', '2'),
                         ('typeboundary_min1=', '6*'+"'periodic'"),
                         ('typeboundary_max1=', '6*'+"'periodic'"),
                         ('typeboundary_min2=', '6*'+"'special'"),
                         ('typeboundary_max2=', '6*'+"'special'")])

meshlist = OrderedDict([('refine_criterion=','3'),
                        ('refine_max_level=', '7'),
                        ('refine_threshold=', '20*0.2d0'),
                        ('derefine_ratio=', '20*1.d0'),
                        ('block_nx1=', '4'),
                        ('block_nx2=', '6'),
                        ('domain_nx1=', '32'),
                        ('domain_nx2=', '24'),
                        ('xprobmin1=', '-2.5d0'),
                        ('xprobmax1=', '2.5d0'),
                        ('xprobmin2=', '0.0d0'),
                        ('xprobmax2=', '3.0d0'),
                        ('stretch_dim(1)=', "'symm'"),
                        ('qstretch_baselevel=','1.1d0'),
                        ('max_blocks=', '12000'),
                        ('nstretchedblocks_baselevel=', '4'),
                        ('ditregrid=', '3')])

paramlist = OrderedDict([('slowsteps=', '100'),
                         ('typecourant=', "'maxsum'"),
                         ('courantpar=', '0.8d0')])

mhd_list = OrderedDict([('mhd_n_tracer=', '1'),
                        ('typedivbfix=', "'glm'"),
                        ('mhd_gravity=', '.T.')])

Drive_type = OrderedDict([('driver_v_pulse=', '.F.'),
                          ('continuous_jet=', '.F.'),
                          ('qt_gaussian_jet=', '.T.')])

sim_parameters = OrderedDict([('driver_amplitude=', '4.0d6 !cm s-1, 1e6cm= 10km'),
                              ('driver_width=', '3.75d7 ! cm, 1e8cm = 1Mm'),
                              ('driver_height=', ' 0.07d0'),
                              ('driver_time=', '300.0d0'),
                              ('bb2=', '50.0d0 !G'),
                              ('tilt_deg=', '0.0d0')])


template = OrderedDict([('&filelist', filelist),
                        ('&savelist', savelist),
                        ('&stoplist', stoplist),
                        ('&methodlist', methodlist),
                        ('&boundlist', boundlist),
                        ('&meshlist', meshlist),
                        ('&paramlist', paramlist),
                        ('&mhd_list', mhd_list),
                        ('&Drive_type', Drive_type),
                        ('&sim_parameters', sim_parameters)])

def parfile_creation(master_dir, par_path, par_name, sav_path, sav_loc, template):
    # Creates dirs for parfile and qsubs
    if os.path.isdir(master_dir+par_path) is False:
        os.makedirs(master_dir+par_path)
    # creates dirs for the location of the save sim op
    if os.path.isdir(sav_loc+sav_path) is False:
        os.makedirs(sav_loc+sav_path)
    template['&filelist']['base_filename='] = "'"+sav_loc+sav_path+'/'+par_name+"_'"
    file = open(master_dir+par_path+'/'+par_name+'.par', 'w')
    for i in template:
        file.write(str(i)+"\n")
        for j in template[i]:
            file.write(str(j)+str(template[i][j])+"\n")
        file.write('/ \n \n')
    file.close()


def submitter_creation(master_dir, par_path, par_name, nb_cores, email, rmem, run_time):
    file_sub = open(master_dir+par_path+'/sub_'+par_name, 'w')
    file_sub.write('#!/bin/bash \n#$ -l h_rt=' + run_time+'\n' +
                   '#$ -pe mpi ' + nb_cores + '\n#$ -l rmem='
                   + rmem +'\n#$ -m ea' + '\n#$ -M ' + email + '\n#$ -j y'
                   + '\n\nmodule load mpi/openmpi/2.1.1/gcc-6.2' 
                   +'\n\nmpirun -np ' +nb_cores+' '+master_dir+'/amrvac -i .'+par_path+'/'+par_name+'.par')
    file_sub.close() 

def submitter_creation_py(master_dir, par_path, par_name, email,
                          rmem, run_time, sav_path, script_path, sav_loc):
    file_sub = open(master_dir+par_path+'/sub_'+par_name+'py', 'w')
    file_sub.write('#!/bin/bash \n#$ -l h_rt=' + run_time+
                   '\n#$ -l rmem=' + rmem + '\n#$ -m ea' + '\n#$ -M '
                   + email + '\n#$ -j y' +
                   '\n\nmodule load apps/ffmpeg/4.1/gcc-4.9.4' +
                   '\nmodule load apps/python/anaconda3-4.2.0' +
                   '\n\nsource activate amrvac' +
                   '\n\npython ' + script_path  + ' ' + sav_loc + sav_path +'/')
    file_sub.close() 

def bash_writter(qsub_names_list):
    file_bash = open('multi_qsub_tilt_sj', 'w')
    file_bash.write('#!/bin/bash\n')
    for i_list in range(len(qsub_names_list)):
        file_bash.write('qsub '+qsub_names_list[i_list]+' &\n')
    file_bash.close()

def bash_movie_maker(qsub_names_list):
    file_bashpy = open('multi_qsub_py', 'w')
    file_bashpy.write('#!/bin/bash\n')
    for i_list in qsub_names_list:
        file_bashpy.write('qsub '+i_list+' &\n')
    file_bashpy.close()


master_dir = '/home/smp16fm/amrvac_v_2_1/amrvac/sim/prominence_Rayleigh_Taylor_2.5D'
os.chdir(master_dir)
# NOTE: need to make sure there is a / infront of shared
sav_loc = '/shared/mhd_jet1/User/smp16fm/j/T'
# corosponding submitters.
run_time = '32:00:00'
nb_cores = '24'
rmem = '0.75G'
email = 'fmackenziedover1@sheffield.ac.uk'

run_time_py = '02:00:00'
rmempy = '2.0G'
email = 'fmackenziedover1@sheffield.ac.uk'
script_path = '/shared/mhd_jet1/User/smp16fm/j/python/animate_sub_plots_sharc.py'



#jet_angle = ['0.0','0.1','0.5','1.0','5.0','10.0','15.0','20.0','25.0','30.0']

## orig setup
#time_driver = ['50', '200', '300']
#bb2 = ['50'] 
#amplitude =  ['40']
#tilt = ['5', '10', '15', '30', '45', '60']


## "standard jet" setup
#time_driver = ['300']
#bb2 = ['60'] 
#amplitude =  ['60']
##tilt = ['0','5', '10', '15', '20']
#tilt = ['25','30', '35', '40', '45', '50', '55', '60']

# "standard jet" setup
time_driver = ['300']
bb2 = ['20', '40', '80', '100'] 
amplitude =  ['20','40','80']
tilt = ['5','15']



## v scan
#time_driver = ['300']
#bb2 = ['50'] 
#amplitude =  ['20', '40', '60', '80']
#tilt = ['30']

qsub_names_list = []

qsubpy_names_list = []
for td in range(len(time_driver)):
    template['&sim_parameters']['driver_time='] = time_driver[td]+'.0d0'
    for b_index in range(len(bb2)):
        template['&sim_parameters']['bb2='] = bb2[b_index]+'.0d0'
        for A_index in range(len(amplitude)):
            template['&sim_parameters']['driver_amplitude='] = amplitude[A_index]+'.0d5'
            for j in range(len(tilt)):
                template['&sim_parameters']['tilt_deg='] = tilt[j]+'.0d0'
                par_name = 'jet_P'+time_driver[td]+'_B'+bb2[b_index]+'_A'+amplitude[A_index]+'_T'+tilt[j]
                par_path =  '/parsj/P'+time_driver[td]+'/B'+bb2[b_index]+'/A'+amplitude[A_index]+'/T'+tilt[j]
                sav_path = '/P'+time_driver[td]+'/B'+bb2[b_index]+'/A'+amplitude[A_index]+'/T'+tilt[j]
                    
                parfile_creation(master_dir, par_path, par_name, sav_path, sav_loc, template)
                
                submitter_creation(master_dir, par_path, par_name, nb_cores, email, rmem, run_time)
                
                submitter_creation_py(master_dir, par_path, par_name, email,
                                      rmempy, run_time_py, sav_path, script_path, sav_loc)
        
                qsub_names_list.append('.'+par_path+'/sub_'+par_name)
                
                qsubpy_names_list.append('.'+par_path+'/sub_'+par_name+'py')
bash_writter(qsub_names_list)
#bash_movie_maker(qsubpy_names_list)


