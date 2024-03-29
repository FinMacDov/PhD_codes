import os
# Taken from https://github.com/Mallcock1/mhd-slab-3d/blob/master/img2vid.py
# ======================================
# Using ffmpeg, we can take the images created for the visualisation and create 
# a video with the research logos overlayed.
#
# The procedure is:
# - define strings which are the ffmpeg command line commands,
# - Use python's os package to run those commands in the command line.
# ======================================

def image2video(filepath=None, prefix='', in_extension='png',
            out_extension='avi', output_name=None, fps=10, n_loops=1, 
            delete_images=False, delete_old_videos=False, res=1080,
            overlay=True, cover_page=False):
                
    if output_name == None:
        output_name = prefix
        
    in_fps = fps
    out_fps = fps
    start_frame = 1
    
# Take input images and uses ffmpeg to make video
    image_2_video = 'ffmpeg -y -framerate '+str(in_fps)+' -start_number '+str(start_frame)+' -i \
    '+os.path.join(filepath, prefix+'%4d.'+in_extension)+' \
    -c:v libx264 -r '+str(out_fps)+' -pix_fmt yuv420p \
    '+os.path.join(filepath, output_name+'.'+out_extension)
    print('comand run for ffmpeg: \n'+image_2_video)

# Delete previous images
    delete_images_cmd = 'DEL "'+os.path.join(filepath, prefix+'*.'+in_extension)+'"'
    if n_loops == 1:
        delete_old_videos_cmd = 'DEL "'+os.path.join(filepath, output_name+'.'+out_extension)+'" \
        "'+os.path.join(filepath, output_name+'_overlay.'+out_extension)+'"'
    else:
        delete_old_videos_cmd = 'DEL "'+os.path.join(filepath, output_name+'_overlay.'+out_extension)+'" "'+os.path.join(filepath, output_name+'_overlay2.'+out_extension)+'"'

# Logo dimensions
    logo_height = str(int(150. * res / 1080.))
    logo_width = str(int(450. * res / 1080.))
    
# Overlay logos
    overlay_image_sp2rc = 'ffmpeg -y -i '+os.path.join(filepath, output_name+'.'+out_extension)+' -i \
    D:\\my_work\\projects\\Asymmetric_slab\\Python\\visualisations\\3d_vis\\sp2rc_logo2.png -filter_complex "[1:0] scale=-1:'+logo_height+' [logo]; \
    [0:0][logo] overlay=main_w-'+logo_width+':main_h-'+logo_height+'" -c:a copy \
    '+os.path.join(filepath, output_name+'_overlay.'+out_extension)

    overlay_image_swat = 'ffmpeg -y -i '+os.path.join(filepath, output_name+'_overlay.'+out_extension)+' -i \
    D:\\my_work\\projects\\Asymmetric_slab\\Python\\visualisations\\3d_vis\\swat_logo2.png -filter_complex "[1:0] scale=-1:'+logo_height+' [logo]; \
    [0:0][logo] overlay=main_w-overlay_w:main_h-'+logo_height+'" -c:a copy \
    '+os.path.join(filepath, output_name+'_overlay2.'+out_extension)

# Coverpage
    cover_page_filename = 'template'
    cover_page_rescale = 'ffmpeg -y -i '+os.path.join(filepath, cover_page_filename+'.'+in_extension)+' -vf scale=1616:866 '+os.path.join(filepath, cover_page_filename+'_coverpage.'+in_extension)    
    cover_page_video = 'ffmpeg -y -framerate '+str(in_fps)+' -i \
    '+os.path.join(filepath, cover_page_filename+'_coverpage.'+in_extension)+' \
    -c:v libx264 -r '+str(out_fps)+' -pix_fmt yuv420p \
    '+os.path.join(filepath, cover_page_filename+'_coverpage.'+out_extension)
    
    if n_loops != 1:
        video_name = os.path.join(filepath, output_name+'_overlay2_looped')
    else:
        video_name = os.path.join(filepath, output_name+'_overlay2')
        
    cover_page_list = "(@echo file '"+os.path.join(filepath, cover_page_filename+'_coverpage.'+out_extension)+"'& @echo file \
'"+video_name+'.'+out_extension+"') > "+os.path.join(filepath, 'cover_page_list.txt')
    
    cover_page_concat = 'ffmpeg -y -f concat -safe 0 -i '+os.path.join(filepath, 'cover_page_list.txt') + ' \
    -c copy '+video_name+'_cp.'+out_extension

# create .txt file to loop through
    loop_list = "(for /l %i in (1,1,"+str(n_loops)+") do @echo file \
    '"+os.path.join(filepath, output_name+'_overlay2.'+out_extension)+"') > "+os.path.join(filepath, 'loop_list.txt')
    
# loop vid n_loops times
    loop = 'ffmpeg -y -f concat -safe 0 -i '+os.path.join(filepath, 'loop_list.txt') + ' -c copy \
    '+os.path.join(filepath, output_name+'_overlay2_looped.'+out_extension)

    # Apply the commands in the command line
    os.system(image_2_video)
    if overlay == True:
        os.system(overlay_image_sp2rc)
        os.system(overlay_image_swat)
    if n_loops != 1:
        os.system(loop_list)
        os.system(loop)
    if cover_page == True:
        os.system(cover_page_rescale)
        os.system(cover_page_video)
        os.system(cover_page_list)
        os.system(cover_page_concat)
    if delete_images == True:
        os.system(delete_images_cmd)
    if delete_old_videos == True:
        os.system(delete_old_videos_cmd)
    
#image2video(prefix='amd_front-top-side_alfven-mixed-driver', output_name='video', 
#            out_extension='mp4', fps=10, n_loops=4, delete_images=False, delete_old_videos=False,
#            cover_page=True)

#img2vid(prefix=prefix, output_name='video', out_extension='mp4', fps=20, n_loops=4, delete_images=True)