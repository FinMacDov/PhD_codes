from shutil import copy
import glob
from pathlib import Path
import os


def coppier(path_list):
    save_dir = path_list[0].split('/')[-2]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for src in path_list:
        dst =  os.path.join(save_dir,src.split('/')[-1])
        copy(src, dst)

test = False

if test == True:
    src_test = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/dummy_file.txt'
    dst_test = 'dummy_file.txt'

    copyfile(src_test, dst_test)

s_path1 = glob.glob('/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/python/*.py')
s_path2 = glob.glob('/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/tilt_python/*.py')

coppier(s_path1)
coppier(s_path2)

