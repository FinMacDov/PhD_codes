from zipfile import ZipFile
from glob import glob
 
root_dir = '/home/fionnlagh/data_jia/jet_P300_B50A_60_' 

path_2_shared_drive = '/run/user/1000/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm/j/python'    
path_2_file = '/jet_P300_B50A_60_'
Full_path = path_2_shared_drive + path_2_file

root_time_folder = glob(root_dir+'/*')
root_time_folder.sort()
csvfiles = glob(root_dir+'/*/*.csv')

for time in root_time_folder:
    csvfiles = glob(time+'/*.csv')
    zipObj = ZipFile(time+'.zip', 'w')
    for csv_file in csvfiles:
        zipObj.write(csv_file,csv_file.split('/')[-1])
    zipObj.close()
