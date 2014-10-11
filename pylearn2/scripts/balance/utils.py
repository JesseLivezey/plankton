import numpy as np
import re
import os
from skimage import transform
from scipy import misc
import shutil

def list_zeros_like(l):
    """Initialize a list of arrays with zeros like the
    arrays in a given list.

    Parameters
    ----------
    l : list
        List of arrays

    Returns
    -------
    new_l : list
        New list with zeros of the same shape as the given list
    """
    new_l = []
    for a in l:
        new_l.append(np.zeros_like(a))

    return new_l

def snapshot_source(src_dir, dest_dir):
    """Snapshot the source code and place it into a directory. This
    is really useful for logging results.

    Parameters
    ----------
    src_dir : string
        Directory path of source files to take a snapshot of

    dest_dir : string
        Destination directory path to store the source files
    """
    snapshot_dir = os.path.join(dest_dir, 'snapshots')
    if os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    shutil.copytree(src_dir, snapshot_dir)

def retrieve_imageset(im_files, size):
    """Retrieve the images from a set of file paths.

    Parameters
    ----------
    im_files : list of strings
        Full file paths of images

    size : sequence of ints
        Size of the image (height, width) to resize to

    Returns
    -------
    img_set : array-like, shape (n_images, height, width, rgb)
        Retrieved image set
    """
    img_set = []
    for im_file in im_files:
        orig_img = misc.imread(im_file).astype('float32')[:,:,:3]/255.
        orig_img = transform.resize(orig_img,size).astype('float32')
        img_set.append(orig_img)

    img_set = np.array(img_set)

    return img_set

def imagify_volume(data_vol):
    """Transform a 3D volume into a 2D image tiled accross the z axis
    which is assumed to be the first dimension
    """
    z,x,y = data_vol.shape
    data_tile = data_vol.reshape(z,x*y)
    tz = int(np.sqrt(z)+1)
    im = tile_raster_images(data_tile, (x,y), (tz,tz),
        scale_rows_to_unit_interval=False,
        output_pixel_vals=False)

    return im

def volumify_response(response, roi_vol, roi_keys):
    """Transform a response vector to a full 3D volume.
    """
    vol = np.zeros_like(roi_vol, dtype='float32')
    for vs_roi in response.keys():
        k = roi_keys[vs_roi]
        vol[roi_vol==k] = response[vs_roi]

    return vol

# From: http://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except:
        return s
     
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def get_nice_filelist(in_dir):
    """Return the file paths of a directory in human expected order.

    Parameters
    ----------
    in_dir : string
        Directory

    Returns
    -------
    filelist : list of strings
        Full paths of files inside in_dir (in human readable order)
    """
    filelist = os.listdir(in_dir)
    filelist = [os.path.join(in_dir, im_file) for im_file in filelist]
    sort_nicely(filelist)
    return filelist

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array