import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import glob
import csv

path_2_shared_drive = '/run/user/1001/gvfs/smb-share:server=uosfstore.shef.ac.uk,share=shared/mhd_jet1/User/smp16fm'    
path_2_file = '/j/2D/P300/B50/A60/'
file_name = 'test_log.csv'
Full_path = path_2_shared_drive + path_2_file + file_name

dt = []
rho = []
m1 = []
m2 = []
e = []
b1 = []
b2 = []

first = True
with open(Full_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if first is True:
            dt.append(row[2])
            rho.append(row[3])
            m1.append(row[4])
            m2.append(row[5])
            e.append(row[6])
            b1.append(row[7])
            b2.append(row[8])
            first = False
        else:
            dt.append(float(row[2]))
            rho.append(float(row[3]))
            m1.append(float(row[4]))
            m2.append(float(row[5]))
            e.append(float(row[6]))
            b1.append(float(row[7]))
            b2.append(float(row[8]))

var = np.array(e[1:-1])
plt.plot(dt[1:-1], var)
plt.show()