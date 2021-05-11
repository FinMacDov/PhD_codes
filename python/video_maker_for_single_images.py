import numpy as np
import glob
from pathlib import Path
import cv2

#movie_name = 'sch_fast'
#file_path = 'jet_P300_B60_A60/t*/*sch*'
#save_dir = 'jet_P300_B60_A60/movie'

#movie_name = 'den'
#file_path = 'jet_P300_B50A_60_/t*/*den*'
#path_2_images = glob.glob(file_path)
#save_dir = 'jet_P300_B50A_60_/movie'

#movie_name = 'rho_Te_sch'
#file_path = 'jet_P300_B60A_60_/figs_for_sj_paper/rho_te_sch/rho*'

#movie_name = 'sj_jet'
#file_path = 'sj_paper/movie/rho_Te_sch/sj_image_for_movie_*.png'

movie_name = 'pscan_jet'
file_path = 'sj_paper/movie/pscan/pscan_for_movie*.png'

path_2_images = glob.glob(file_path)
path_2_images = np.asarray(path_2_images)
list_of_inds = []
for p_name in path_2_images:
#    list_of_inds.append(int(p_name.split('_')[-1].split('.')[0]))
    list_of_inds.append(int(p_name.split('_')[-1].split('.')[0].split('e')[-1]))
fixed_indexs = np.argsort(list_of_inds)
path_2_images = path_2_images[fixed_indexs]
#save_dir = 'image_check_sj/movie'
save_dir = 'sj_paper/movie'
# stupid error where 10 is first, will need to rename files
# For these purpose skipping one time step is fine
Path(save_dir).mkdir(parents=True, exist_ok=True)    
fps = 16 # 2
in_extension='png'
out_extension='avi'

img_array = []
for filename in path_2_images:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(save_dir+'/'+movie_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
