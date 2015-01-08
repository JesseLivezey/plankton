import os, sys, h5py
from scipy.ndimage import imread
import numpy as np

if len(sys.argv) < 3:
    print "Usage: python gen_train_h5.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

# Get list of classes and create label dictionary
classes = os.listdir(fi)
classes = sorted(classes)
class_label_dict = {}
for ii,cls in enumerate(classes):
    class_label_dict[cls] = ii

# Determine number of files
n_files = 0
for cls in classes:
    imgs = os.listdir(os.path.join(fi,cls))
    n_files += len(imgs)

# Determine image size
sx, sy = imread(os.path.join(fi,cls,imgs[0])).shape

# Write images into single array
im_array = np.zeros((n_files,sx,sy), dtype='float32')
class_array = np.zeros(n_files, dtype='int')
image_id = np.zeros(n_files, dtype='int')
ii = 0
for cls in classes:
    imgs = sorted(os.listdir(os.path.join(fi,cls)))
    for img in imgs:
        name = int(img.split('.')[0])
        data = imread(os.path.join(fi,cls,img))
        assert data.shape == (sx,sy), os.path.join(fi,cls,img)+\
                                     ' is of the wrong shape'+\
                                     ' got '+str(data.shape)+\
                                     ' expected '+str((sx,sy))
        im_array[ii] = data
        class_array[ii] = class_label_dict[cls]
        image_id[ii] = name
        ii += 1

print im_array.min()
print im_array.max()

# Write output file
os.chdir(fo)
with h5py.File('train.h5','w') as f:
    f.create_dataset('X', data=im_array)
    f.create_dataset('y', data=class_array)
    f.create_dataset('id', data=image_id)
