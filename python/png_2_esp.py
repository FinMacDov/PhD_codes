from PIL import Image
from glob import glob
 
root_dir = '/home/fionnlagh/Documents/PhD/blob_images' 
 
pngfiles = glob(root_dir+'/*/*.png')
for u in pngfiles:
    image_png = u
    out = u.replace('png', 'eps')
    image_png = u
    im = Image.open(image_png)
    fig = im.convert('RGB')
    fig.save(out, lossless=True)