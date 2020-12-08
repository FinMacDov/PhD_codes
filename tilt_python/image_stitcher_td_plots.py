import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as Path_creator
import glob


def gallery(array, ncols=4):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def make_array(image_name):
    from PIL import Image
    first = True
    for img in image_name:
#        print(img[0])
        if first:
            first = False
            image = np.array([np.asarray(Image.open(img[0]))])
        else:
            image = np.vstack((image, np.array([np.asarray(Image.open(img[0]))])))
    return image

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm' 

name = [ 'jet_P300_B60_A60_T5', 
        'jet_P300_B60_A60_T10', 'jet_P300_B60_A60_T15',
        'jet_P300_B60_A60_T20', 'jet_P300_B60_A60_T25',
        'jet_P300_B60_A60_T30', 'jet_P300_B60_A60_T35',
        'jet_P300_B60_A60_T40', 'jet_P300_B60_A60_T45',
        'jet_P300_B60_A60_T50', 'jet_P300_B60_A60_T55',
        'jet_P300_B60_A60_T60']

HoI = 1
save_dir = 'sharc_run/fig_for_paper/'
Path_creator(save_dir).mkdir(parents=True, exist_ok=True)

list_of_dirs = []
record_img_size = True
for name_path in name:
    dir_2_image = glob.glob(path_2_shared_drive+'/j/tilt_python/sharc_run/sj_td_plot_sfigs/'+name_path+'/'+str(int(HoI))+'Mm/*.png')
    list_of_dirs.append(dir_2_image)

f, ax = plt.subplots()
f.set_size_inches(32, 18)

stack_of_images = make_array(list_of_dirs)
result = gallery(stack_of_images)
plt.axis('off')
plt.imshow(result)
plt.show()
#plt.savefig(save_dir+name[idx]+marker+'.png', bbox_inches='tight',
#            format='png', dpi=800, pad_inches = 0)
