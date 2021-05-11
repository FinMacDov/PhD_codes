import sys
import os
import glob
import numpy as np
import shutil

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm' 

name = ['jet_P300_B60A_60_','jet_P300_B60_A60']
indexs_of_interest = np.linspace(1,23,8,dtype=int)

list_of_dirs = []
t_folder_names = []
sch_list =  np.asarray(glob.glob(path_2_shared_drive+'/j/python/'+name[-1]+'/*/T_*__Slice_z_numerical_schlieren.png'))

delete_zoom_names = []
for sch_idx, sch_name in enumerate(sch_list):
    if 'zoomed' in sch_name:
        delete_zoom_names.append(sch_idx)
sch_list = np.delete(sch_list, delete_zoom_names)

for idx, name_path in enumerate(name):
    dummy_t = []
    dir_2_image = path_2_shared_drive+'/j/python/'+name_path
    dummy_list_of_dirs = np.asarray(glob.glob(dir_2_image+'/t*'))
    list_of_dirs.append(dummy_list_of_dirs)
    sch_list
    for item in list_of_dirs[idx]:
        dummy_t.append(os.path.split(item)[-1])
    if idx ==0:
        for i, item in enumerate(dummy_t):
            dummy_t[i] = item.replace('_','')
    t_folder_names.append(np.asarray(dummy_t))  

index_of_matches = []
for i in range(len(t_folder_names[0])):
    name_matcher = t_folder_names[0][i]
    for j in range(len(t_folder_names[1])):
        name_check = t_folder_names[1][j]
#        print( name_matcher,name_check,name_matcher==name_check)
        if name_matcher==name_check:
            index_of_matches.append(j)
            pass

for nb, item in enumerate(sch_list[index_of_matches]):
    sch_names = os.path.split(item)[-1]
#    output_names = (list_of_dirs[0][nb+1] + '/' + sch_names)
    output_names = (list_of_dirs[0][nb] + '/' + sch_names)
    print(sch_list[index_of_matches][nb], output_names)
    shutil.copyfile(sch_list[index_of_matches][nb], output_names)    


