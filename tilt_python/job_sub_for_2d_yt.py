#import sys
#import os
import numpy as np
import glob
from pathlib import Path
#import pandas as pd

def bash_writter(qsub_names_list):
    file_bash = open('data_gather_tilt_qsub', 'w')
    file_bash.write('#!/bin/bash\n')
    for i_list in range(len(qsub_names_list)):
        file_bash.write('qsub '+qsub_names_list[i_list]+' &\n')
    file_bash.close()


def submitter_creation_py(path_2_shared_drive, path_2_sub_dir, file_name, email,
                          rmem, run_time, script_path, var_name):
    Path(path_2_shared_drive+path_2_sub_dir).mkdir(parents=True, exist_ok=True)
#    print('I have created: '+path_2_shared_drive+path_2_sub_dir)
    fname = path_2_shared_drive+path_2_sub_dir+'/yt_plots_for_paper_'+file_name+'_py.sh'
    file_sub = open(fname, 'w')
    file_sub.write('#!/bin/bash \n#$ -l h_rt=' + run_time+
                   '\n#$ -l rmem=' + rmem + '\n#$ -m ea' + '\n#$ -M '
                   + email + '\n#$ -j y' +
                   '\nexport given_path=' + var_name +
                   '\nexport given_jet_name=' + '"' + jet_name + '"' +
                   '\nmodule load apps/python/anaconda3-4.2.0' +
                   '\n\nsource activate amrvac' +
                   '\n\npython ' + script_path)
    file_sub.close() 
    return fname


path_2_shared_drive = '/shared/mhd_jet1/User/smp16fm/j'    
dir_paths =  glob.glob(path_2_shared_drive+'/T/P300/B60/A60/T*')
# testing 
#dir_paths = [dir_paths[0]]


script_path = '/shared/mhd_jet1/User/smp16fm/j/tilt_python/2d_yt_plots_for_paper_sharc.py'
path_2_sub_dir = '/yt_plot_sub_dir_tilt' 
email = 'fmackenziedover1@sheffield.ac.uk'
rmem = '4.0G'
run_time = '96:00:00'
#run_time = '00:10:00'

list_of_fnames = []
for path in dir_paths:
    path_parts = path.split('/')
    path_parts = path_parts[-4:]
    path_numerics = np.zeros(len(path_parts))
    Fi = True

    # join names parts in desired form
    for item in path_parts:
        if Fi == True:
            Fi = False
            jet_name = 'jet_'+item
        else:
            jet_name += '_'+item 
    var_name = '"'+path+'"'
    list_of_fnames.append(submitter_creation_py(path_2_shared_drive, path_2_sub_dir, jet_name, email,
                          rmem, run_time, script_path, var_name))
                          
bash_writter(list_of_fnames)
  