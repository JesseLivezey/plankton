import os, sys
from scipy.misc import imread, imsave
from scipy.ndimage.interpolation import rotate
import numpy as np
from sklearn import mixture
np.random.seed(1)

#can definitely cut down number of variables to store the image
#in the for loop, but liked
#the convention to keep track of what operations had been done on the image
#based on name.  f = flip, r=rotated, s=shifted; used function composition
#convention


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

        g = mixture.GMM(n_components=1, covariance_type = 'full')
        g.fit(xy_pts.T) 
        gmean = g.means_
        gcov = g.covars_
        u,s,v = np.linalg.svd(gcov[0],full_matrices=False)
        orig_angle = np.arctan(u[1,0]/u[0,0])*180./np.pi
        
        #now use the rotation matrix from the SVD of the correlation matrix to 
        #rotate the image and the xy_pts
        roimg = rotate(orig_img, 360-orig_angle, reshape=True, cval=orig_img.max())
        rad_angleRot = (360-orig_angle)*np.pi/180.
        rotMatrix = np.array([[np.cos(rad_angleRot), -np.sin(rad_angleRot)], 
                                 [np.sin(rad_angleRot),  np.cos(rad_angleRot)]])

        rotxy_pts = np.dot(rotMatrix,xy_pts)

        #am a little bit worried that the rotation has moved the center a bit, so doing another
        #GMM on the rotated data.  Can probably take this out if it's too slow.  Would normally
        #shift and rotated, but shifting first on some images causes a pacman effect on the other
        #side of the image because of the roll function I'm using.  
        g.fit(rotxy_pts.T)
        gmean = g.means_

        #now will shift the image based to center the gaussian
        srotxy_pts = np.zeros_like(rotxy_pts)
        srotxy_pts[0,:] = rotxy_pts[0,:]-int(gmean[0,0])
        srotxy_pts[1,:] = rotxy_pts[1,:]-int(gmean[0,1])

        #now fill flip the image so any top/bottom heavy features will be on the top and any assymetry
        #in the left right direction will be flipped so that it's always on the right.
        fsroimg = sroimg
        if np.sum(srotxy_pts[0,:]**3) < 0:
            fsroimg = fsroimg[:,::-1]
            #print 'flip across y axis'
        if np.sum(srotxy_pts[1,:]**3) < 0:
            fsroimg = fsroimg[::-1,:]
            #print 'flip across x axis'  
        
        imsave(os.path.join(fo,cls,img), fsroimg)