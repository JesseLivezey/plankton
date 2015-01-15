"""
Plankton dataset wrapper for semi-supervised networks.
"""
__authors__ = "Jesse Livezey"

import numpy as N
np = N
import cPickle, h5py, os
from theano.compat.six.moves import xrange
from pylearn2.datasets import semi_supervised, dense_design_matrix
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import serial

class SemiPlankton(semi_supervised.SemiSupervised):
    """
    The National Data Science Bowl dataset

    Parameters
    ----------
    folder : str
        Folder which contains data files.
    which_set : str
        'train', 'valid', or 'test'
    seed : int
    """

    def __init__(self, folder, which_set, seed):
        self.args = locals()
        rng = np.random.RandomState(seed)
        self.rng = make_np_rng(rng, which_method=['permutation'])

        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')

        folder = serial.preprocess(folder)

        with open(os.path.join(folder,'label_mapping.pkl'),'r') as f:
            self.label_mapping = cPickle.load(f)

        with h5py.File(os.path.join(folder,'train.h5'), 'r') as f:
            topo_view = f['X'].value[...,np.newaxis]/255.
            y = f['y'].value
            self.ids = f['id'].value
        n_examples = topo_view.shape[0]
        perm = rng.permutation(n_examples)
        topo_view = topo_view[perm]
        y = y[perm]
        with h5py.File(os.path.join(folder,'test.h5'), 'r') as f:
            self.unlabeled = f['X'].value[...,np.newaxis]/255.
            self.ids_unlabeled = f['id'].value
        split = {'train': .8,
                 'valid': .1,
                 'test': .1}
        assert np.all(np.array(split.values()) > 0.)
        assert np.allclose(np.sum(split.values()), 1.)
        n_test = int(split['test']*n_examples)
        n_valid = int(split['valid']*n_examples)
        n_train = n_examples-n_test-n_valid

        train_topo_view = topo_view[:n_train]
        if which_set == 'train':
            topo_view = train_topo_view
            y = y[:n_train]
        elif which_set == 'valid':
            topo_view = topo_view[n_train:n_train+n_valid]
            y = y[n_train:n_train+n_valid]
        else:
            topo_view = topo_view[n_train+n_valid:]
            y = y[n_train+n_valid:]
        y = y[...,np.newaxis]

        self.feature_mean = train_topo_view.mean(0)
        topo_view -= self.feature_mean
        y_labels = max(self.label_mapping.values())+1
        axes = ['b',0,1,'c']
        view_converter = dense_design_matrix.DefaultViewConverter(topo_view.shape[1:], axes=axes)

        L = topo_view.reshape(-1, np.prod(topo_view.shape[1:]))
        unlabeled = self.unlabeled.reshape(-1, np.prod(self.unlabeled.shape[1:]))
        if which_set == 'train':
            X = L
        else:
            X = np.vstack((L, unlabeled))

        super(SemiPlankton, self).__init__(X=X, V=X, L=L, y=y,
                                    view_converter=view_converter,
                                    y_labels=y_labels)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        return SemiPlankton(**args)

    def get_valid_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'valid'
        return SemiPlankton(**args)

