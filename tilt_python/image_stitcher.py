import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path as Path_creator
import os
import glob

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm' 

#name = ['jet_P300_B50_A40_T5_', 'jet_P300_B50_A40_T10_',
#        'jet_P300_B50_A40_T15_']
#name = ['jet_P300_B50_A40_T30_', 'jet_P300_B50_A40_T45_', 
#        'jet_P300_B50_A40_T60_']

#name = ['jet_P300_B50_A60_T30_', 'jet_P300_B50_A80_T30_', 
#        'jet_P300_B50_A40_T60_']

name = ['jet_P300_B60_A60_T0_', 'jet_P300_B60_A60_T5_', 
        'jet_P300_B60_A60_T10_', 'jet_P300_B60_A60_T15_']
#        'jet_P300_B60_A60_T20_']


#indexs_of_interest = np.array([10,30,50,70,90,110])

indexs_of_interest = np.array([10,30,60,80,110,129])

#indexs_of_interest = np.array([70,90,110])
#indexs_of_interest = np.linspace(1,23,8,dtype=int)

#indexs_of_interest = np.linspace(1,23,8,dtype=int)
nb_of_images_per_plot = 4
save_dir = '/tilt_python/sharc_run/yt_images/fig_for_pres'
#save_dir = 'fig_for_paper/'
Path_creator(save_dir).mkdir(parents=True, exist_ok=True)
list_of_dirs = []
for name_path in name:
#    dir_2_image = path_2_shared_drive+'/j/tilt_python/'+name_path
    dir_2_image = path_2_shared_drive+'/j/tilt_python/sharc_run/yt_images'+name_path
    dummy_list_of_dirs = np.asarray(glob.glob(dir_2_image+'/*'))
    list_of_dirs.append(dummy_list_of_dirs[indexs_of_interest])
set_cound=1
# creates multiple images
two_by_n = False
# creates single image
n_by_two = False
# I may have broken earlier parts of the code
# for 3 vars images
n_by_three = False

n_by_four = True

#word_search_1 = 'den'
#word_search_2 = 'te'
#word_search_3 = 'sch'

word_search_1 = 'den'
word_search_2 = word_search_1
word_search_3 = word_search_1

number_of_vstacks = 1
side_stack = [] 
h_stack = []

image_stamp = 0
for idx, dir_name in enumerate(list_of_dirs[0]):
    marker = 't' 
    if len(list_of_dirs)==1:
        file_names = glob.glob(dir_name+'/*')
        image1 = plt.imread([s for s in file_names if word_search_1 in s][0])
        image2 = plt.imread([s for s in file_names if word_search_2 in s][0])
        image3 = plt.imread([s for s in file_names if word_search_3 in s][0])
    else:
        file_names1 = glob.glob(dir_name+'/*')
        file_names2 = glob.glob(list_of_dirs[1][idx]+'/*')
        file_names3 = glob.glob(list_of_dirs[2][idx]+'/*')
        file_names4 = glob.glob(list_of_dirs[3][idx]+'/*')
        image1 = plt.imread([s for s in file_names1 if word_search_1 in s][0])
        image2 = plt.imread([s for s in file_names2 if word_search_1 in s][0])
        image3 = plt.imread([s for s in file_names3 if word_search_1 in s][0])
        image4 = plt.imread([s for s in file_names4 if word_search_1 in s][0])
    if n_by_two == True:
        h_stack.append(np.concatenate((image1, image2), axis=0))
        if idx==len(list_of_dirs[0])-1:
            side_stack = np.concatenate((h_stack[0],h_stack[1]), axis = 1)
            for i in range(2,len(h_stack)):
                side_stack = np.concatenate((side_stack,h_stack[i]), axis = 1)
            plt.axis('off')
            plt.imshow(side_stack)
            for mark in indexs_of_interest:
                marker +=  '_'+str(mark)
            plt.savefig(save_dir+name[0]+'_'+marker+'.png', bbox_inches='tight',
                        format='png', dpi=800, pad_inches = 0)
    if two_by_n == True:
        # h stcks images and saves them
        side_stack.append(np.concatenate((image1, image2), axis=1))
        # v stacks the images and wipes it face
        if idx>0 and (idx+1)%(number_of_vstacks) ==0:
            low_stack = np.concatenate((side_stack[0],side_stack[1]), axis = 0)
            for i in range(2, len(side_stack)):
                low_stack = np.concatenate((low_stack, side_stack[i]), axis = 0)
            plt.axis('off')
            plt.imshow(low_stack)
            for mark in indexs_of_interest:
                marker +=  '_'+str(mark)
            plt.savefig(save_dir+name[0]+'_'+marker+'.png', bbox_inches='tight',
                        format='png', dpi=800, pad_inches = 0)
            side_stack = [] 
            
    if n_by_three == True:
        h_stack.append(np.concatenate((image1, image2, image3), axis=0))
        if (idx+1)%nb_of_images_per_plot==0:
            side_stack = np.concatenate((h_stack[0],h_stack[1]), axis = 1)
            for i in range(2,len(h_stack)):
                side_stack = np.concatenate((side_stack,h_stack[i]), axis = 1)
            plt.axis('off')
            ll = set_cound*(nb_of_images_per_plot-1)
            ul = ll+set_cound+nb_of_images_per_plot
            for mark in indexs_of_interest[ll:ul]:
                marker +=  '_'+str(mark)
            print(marker)
            plt.imshow(side_stack)
            image_stamp += 1
            plt.savefig(save_dir+name[0]+marker+'.png', bbox_inches='tight',
                        format='png', dpi=800, pad_inches = 0)
            print('I made an image!!!', save_dir+name[0]+marker+'.png')
            set_cound += 1
            h_stack = []

    if n_by_four == True:
        h_stack.append(np.concatenate((image1, image2, image3, image4), axis=0))
        if (idx+1)%nb_of_images_per_plot==0:
            side_stack = np.concatenate((h_stack[0],h_stack[1]), axis = 1)
            for i in range(2,len(h_stack)):
                side_stack = np.concatenate((side_stack,h_stack[i]), axis = 1)
            plt.axis('off')
            ll = set_cound*(nb_of_images_per_plot-1)
            ul = ll+set_cound+nb_of_images_per_plot
            for mark in indexs_of_interest[ll:ul]:
                marker +=  '_'+str(mark)
            print(marker)
            plt.imshow(side_stack)
            image_stamp += 1
            plt.savefig(save_dir+name[0]+marker+'.png', bbox_inches='tight',
                        format='png', dpi=800, pad_inches = 0)
            print('I made an image!!!', save_dir+name[0]+marker+'.png')
            set_cound += 1
            h_stack = []
