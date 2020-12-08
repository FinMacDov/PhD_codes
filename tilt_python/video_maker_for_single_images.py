import numpy as np
import glob
from pathlib import Path
import cv2
import os

#movie_name = 'sch_fast'
#file_path = 'jet_P300_B60_A60/t*/*sch*'
#save_dir = 'jet_P300_B60_A60/movie'

rename = False

fps = 6 # 2
in_extension='png'
out_extension='avi'

#movie_name = 'widths'
#search_image_word = '*'
#search_file_word = 'image_check/jet_P300_B60_A60_T0/'

#movie_name = 'run'
#search_image_word = '/t_*/*run*'
#search_file_word = 'jet_P200_B50_A40_T15_'

#movie_name = 'den_mov'
#search_image_word = '/t_*/*den*'
#search_file_word = 'jet_P300_B60_A60_T*'
movie_name = 'width_track_mov'
search_image_word = '/*'
search_file_word = 'jet_P300_B60_A60_T*'

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
# 2D
#path_root = '/j/tilt_python/sharc_run/yt_images/'
path_root = '/j/tilt_python/sharc_run/image_check/'
file_paths = glob.glob(path_2_shared_drive+path_root+search_file_word)

for file in file_paths:
    file_path = file+search_image_word
    if rename == True:
        dummy_path_2_images = glob.glob(file_path)
        for dummy_file in dummy_path_2_images:
            dummy_path_parts = dummy_file.split('/')
            dummy_last_part = dummy_path_parts[-1].split('_')
            dummy_num_part = dummy_last_part[-1].split('.')
            dummy_num_part[0] = dummy_num_part[0].zfill(4)
            dummy_last_part[-1] = '.'.join(dummy_num_part)
            dummy_path_parts[-1] = '_'.join(dummy_last_part)
            new_name = '/'.join(dummy_path_parts)
            os.rename(dummy_file,new_name)

    path_2_images = glob.glob(file_path)
    number_list = []
    for fname in path_2_images:
        number_list.append(int(fname.split('_')[-1].split('.')[0]))
#        number_list.append(int(fname.split('_')[-4][-4:]))
    sorted_index = np.argsort(number_list)
    path_2_images = np.asarray(path_2_images)[sorted_index]
    
    jet_name = file_path.split('/')[-2]
#    jet_name = file_path.split('/')[-3]
    save_dir = 'sharc_run/yt_images/movies/'+jet_name
    Path(save_dir).mkdir(parents=True, exist_ok=True)    
    
    img_array = []
    for filename in path_2_images:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(save_dir+'/'+jet_name+movie_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
