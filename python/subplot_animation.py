import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

sys.path.append("/home/smp16fm/forked_amrvac/amrvac/tools/python")

from amrvac_pytools.datfiles.reading import amrvac_reader
from amrvac_pytools.vtkfiles import read, amrplot

def subplot_animation(path2files, save_dir, dummy_name='', refiner=None,
                      text_x_pos=0.85, text_y_pos=0.01, time_start=0,
                      time_end=None, start_frame=0, fps=1, in_extension='png',
                      out_extension='avi'):
    '''
    For making movies with subplotting using polyplot for vtk files.
    Inputs:
     path2files - (str) give path to files (use * to select multple files).
     dummy name - (str) useful for picking out particular file names.
     refiner - (str) will remove paths contain str put in here.
     text_x_pos and text_y_pos - (float) location of time on plots.
     time_start - (int) starting point for reading vtk files.
     start_frame - (int) first frame for ffmpeg.
     fps - (int) frames per second.
     in_extension - (str) expects input to png.
     out_extension - (str) decides files type the movie is.
     save_dir - (str) save location of the images and movies
    '''
    var_names = ["rho",
                 "v1",
                 "v2",
                 "p",
                 "b1",
                 "b2",
                 "trp1",
                 "T",
                 "Cs",
                 "beta",
                 "sch",
                 "e"]

    function = [
        lambda x: x.rho,
        lambda x: x.v1,
        lambda x: x.v2,
        lambda x: x.p,
        lambda x: x.b1,
        lambda x: x.b2,
        lambda x: x.trp1,
        lambda x: x.T,
        lambda x: x.Cs,
        lambda x: x.beta,
        lambda x: x.sch,
        lambda x: x.en,
    ]

    cmaps_colours = ['gist_heat',
                     'seismic',
                     'seismic',
                     'BuGn',
                     'seismic',
                     'hot',
                     'inferno',
                     'coolwarm',
                     'copper',
                     'bone',
                     'binary',
                     'BuGn']

    list_of_names = []
    list_of_indexs = []
    list_of_paths = []
    list_of_full_dummy_paths = glob.glob((path2files+dummy_name+'*0000.vtu'))
#    print(list_of_full_dummy_paths)
    # removes any array element that contains refiner terms.
    if refiner is not None:
        indexs_to_remove = [ind for ind, it in enumerate(list_of_full_dummy_paths) if refiner in it]
        shift = 0
        for indx in indexs_to_remove:
            # removes element from array
            del list_of_full_dummy_paths[indx+shift]
            # adject postion based on new array
            shift -= 1

    for indx, item in enumerate(list_of_full_dummy_paths):
        list_of_names.append(item.split('/')[-1])
        dummy_path = item.split('/')[0:-1]
        list_of_paths.append(os.path.join(*dummy_path)+'/')

    for j in range(len(list_of_names)):
        name = list_of_names[j][0:-8]  # -8 clips 0000.vtu
        path2save_images = save_dir+'/'+name+"/images"
        path2save_movies = save_dir+'/'+name+"/movies"
        filename = '/'+list_of_paths[j] + name
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            # directory already exists
            pass
        try:
            os.makedirs(path2save_images)
            os.makedirs(path2save_movies)
        except FileExistsError:
            # directory already exists
            pass

        # loads for time step
        ds = read.load_vtkfile(5, file=filename, type='vtu')
        ds0 = read.load_vtkfile(0, file=filename, type='vtu')
        rho_range = [min(ds.rho), max(ds.rho)]
        maxv1 = np.max(abs(ds.v1))
        maxv1 -= 0.2*maxv1
        maxv2 = np.max(abs(ds.v2))
        maxv2 += 0.5*maxv2
        v1_range = [-maxv1, maxv1] # [-6.5e6, 6.5e6] #   
        v2_range = [-maxv2, maxv2] # [-8e6, 8e6] #   
        p_range = [min(ds.p), max(ds.p)]
        maxb1 = 0.05*np.max(abs(ds.b1))
#        maxb1 -= 0.1*maxv1
        maxb2 = np.max(abs(ds0.b2))
#        maxb2 -= 0.1*maxv1
        b1_range = [-maxb1, maxb1] # [-60, 60] #  
        b2_range = [-0.5*maxb2, maxb2] # [-60, 60] # 
        trp1_range = [0, 100]
        T_range = [8e3, 2e6]
        Cs_range = [min(ds.Cs), max(ds.Cs)]
        beta_range = [min(ds.beta), max(ds.beta)]
        sch_range = [min(ds.sch), max(ds.sch)]
        en_range = [min(ds.en), max(ds.en)]

        cmap_ranges = [
            rho_range,
            v1_range,
            v2_range,
            p_range,
            b1_range,
            b2_range,
            trp1_range,
            T_range,
            Cs_range,
            beta_range,
            sch_range,
            en_range,
        ]
        if time_end is None:
            number_of_files = len(glob.glob(list_of_full_dummy_paths[j][0:-7]+'*.vtu'))
        else:
            number_of_files = time_end

        for k in range(time_start, number_of_files):
            ds = read.load_vtkfile(k, file=filename, type='vtu')
            fig, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24), (ax31, ax32, ax33, ax34)) = plt.subplots(nrows=3, ncols=4,figsize=(18,10))
            axis_list = [ax11, ax12, ax13, ax14,
                         ax21, ax22, ax23, ax24,
                         ax31, ax32, ax33, ax34]
            for i in range(len(axis_list)):
                p1 = amrplot.polyplot(function[i](ds),
                                      ds,
                                      clear=False,
                                      fig=fig,
                                      axis=axis_list[i],
                                      min=cmap_ranges[i][0],
                                      max=cmap_ranges[i][-1],
                                      orientation="vertical",
                                      function=function[i],
                                      cmap=cmaps_colours[i],
                                      title=var_names[i],
                                      yrange=[0, 1e9],
                                      xrange=[-6e8, 6e8],
                                      log_info=False
                                      )
            spacer = 0.4
            plt.subplots_adjust(wspace=spacer, hspace=spacer)
            time = ds.time
            time_text = 'Time: '+str(round(time, 2))+' s'
            fig.text(text_x_pos, text_y_pos, time_text, size=14)
        #    mng = plt.get_current_fig_manager()
        #    mng.resize(*mng.window.maxsize())
        #    plt.show()
            plt.savefig(path2save_images+'/'+name+str(k).zfill(4)+'.png')
            plt.clf()

#        image_2_video = 'ffmpeg -y -framerate '+str(fps)+' -start_number '+str(start_frame)+' -i \
#        '+os.path.join(path2save_images, name+'%4d.'+in_extension)+' \
#        -c:v libx264 -r '+str(fps)+' -pix_fmt yuv420p \
#        '+os.path.join(path2save_movies, name+'.'+out_extension)

        image_2_video = 'ffmpeg -framerate '+str(fps)+' -i \
                        '+os.path.join(path2save_images, name+'%4d.'+in_extension)+' -y \
                        '+os.path.join(path2save_movies, name+'.'+out_extension)


        print(image_2_video)
        os.system(image_2_video)
