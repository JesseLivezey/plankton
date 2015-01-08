import os, sys, h5py
from scipy.ndimage import imread
import numpy as np

if len(sys.argv) < 3:
    print "Usage: python gen_test_h5.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

# Get list of files
imgs = sorted(os.listdir(fi))

# Determine number of files
n_files = len(imgs)

# Determine image size
sx, sy = imread(os.path.join(fi,imgs[0])).shape

# Write images into single array
im_array = np.zeros((n_files,sx,sy), dtype='float32')
image_id = np.zeros(n_files, dtype='int')
ii = 0
for img in imgs:
    name = int(img.split('.')[0])
    data = imread(os.path.join(fi,img))
    assert data.shape == (sx,sy), os.path.join(fi,img)+\
                                 ' is of the wrong shape'+\
                                 ' got '+str(data.shape)+\
                                 ' expected '+str((sx,sy))
    im_array[ii] = data
    image_id[ii] = name
    ii += 1

# Write output file
os.chdir(fo)
with h5py.File('test.h5','w') as f:
    f.create_dataset('X', data=im_array)
    f.create_dataset('id', data=image_id)
