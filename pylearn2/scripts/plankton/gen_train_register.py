import os, sys
from scipy.misc import imread, imsave
from scipy.ndimage.interpolation import rotate
import numpy as np

if len(sys.argv) < 3:
    print "Usage: python gen_train_register.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(os.path.join(fi,cls))
    for img in imgs:
        orig_img = imread(os.path.join(fi,cls,img))
        nimg = 1. - orig_img/255.
        # orig_angle = np.arctan((nimg.shape[0]*1.)/nimg.shape[1])*180./np.pi
        # n_angles = np.round(orig_angle/45.)
        # oimg = rotate(nimg, 360-(45*n_angles), reshape=True, mode='nearest')
        pts = np.array(np.where(nimg >= np.median(nimg[nimg>0.])))
        xy_pts = np.zeros_like(pts, dtype='float')
        xy_pts[0] = pts[1] - nimg.shape[1]/2.
        xy_pts[1] = nimg.shape[0]/2. - pts[0]

        u,s,v = np.linalg.svd(xy_pts,full_matrices=False)
        orig_angle = np.arctan(u[1,0]/u[0,0])*180./np.pi
        # orig_angle = np.abs(orig_angle)
        oimg = rotate(orig_img, 360-orig_angle, reshape=True, cval=orig_img.max())
        imsave(os.path.join(fo,cls,img), oimg)

