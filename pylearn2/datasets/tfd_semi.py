"""
.. todo::

    WRITEME
"""
import numpy as np
from pylearn2.datasets.semi_supervised import SemiSupervised
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan


class TFDSemi(SemiSupervised):
    """
    Pylearn2 wrapper for the Toronto Face Dataset.
    http://aclab.ca/users/josh/TFD.html

    Parameters
    ----------
    which_set : str
        Dataset to load. One of ['train','valid','test','unlabeled'].
    fold : int in {0,1,2,3,4}
        TFD contains 5 official folds for train, valid and test.
    image_size : int in [48,96]
        Load smaller or larger dataset variant.
    example_range : array_like or None, optional
        Load only examples in range [example_range[0]:example_range[1]].
    center : bool, optional
        Move data from range [0., 255.] to [-127.5, 127.5]
        False by default.
    scale : bool, optional
        Move data from range [0., 255.] to [0., 1.], or
        from range [-127.5, 127.5] to [-1., 1.] if center is True
        False by default.
    shuffle : WRITEME
    one_hot : WRITEME
    rng : WRITEME
    seed : WRITEME
    preprocessor : WRITEME
    axes : WRITEME
    """

    mapper = {'unlabeled': 0, 'train': 1, 'valid': 2, 'test': 3,
            'full_train': 4, 'semisupervised': 5}

    def __init__(self, which_set, fold=0, image_size=48,
                 center=False, scale=False,
                 shuffle=False, one_hot=False, rng=None, seed=132987,
                 preprocessor=None, axes=('b', 0, 1, 'c')):

        if which_set not in self.mapper.keys():
            raise ValueError("Unrecognized which_set value: %s. Valid values" +
                             "are %s." % (str(which_set),
                                          str(self.mapper.keys())))
        assert (fold >= 0) and (fold < 5)

        self.args = locals()

        # load data
        path = '${PYLEARN2_DATA_PATH}/faces/TFD/'
        if image_size == 48:
            data = load(path + 'TFD_48x48.mat')
        elif image_size == 96:
            data = load(path + 'TFD_96x96.mat')
        else:
            raise ValueError("image_size should be either 48 or 96.")

        # retrieve indices corresponding to `which_set` and fold number
        if self.mapper[which_set] == 4:
            feature_indices = (data['folds'][:, fold] == 1) + \
                          (data['folds'][:, fold] == 2)
            labeled_indices = feature_indices
        elif self.mapper[which_set] == 5:
            labeled_indices = data['folds'][:, fold] == 1
            unlabeled_indices = data['folds'][:, fold] == 0
            feature_indices = np.logical_or(labeled_indices, unlabeled_indices)
        else:
            feature_indices = data['folds'][:, fold] == self.mapper[which_set]
            labeled_indices = feature_indices
        assert labeled_indices.sum() > 0

        # get images and cast to float32
        features = data['images'][feature_indices].astype('float32')
        features = features.reshape(-1, image_size ** 2)

        labeled = data['images'][labeled_indices].astype('float32')
        labeled = labeled.reshape(-1, image_size ** 2)

        if center and scale:
            features[:] -= 127.5
            features[:] /= 127.5
            labeled[:] -= 127.5
            labeled[:] /= 127.5
        elif center:
            features[:] -= 127.5
            labeled[:] -= 127.5
        elif scale:
            features[:] /= 255.
            labeled[:] /= 255.

        if shuffle:
            rng = make_np_rng(rng, seed, which_method='permutation')
            rand_idx = rng.permutation(len(features))
            features = features[rand_idx]
            rand_idx_labeled = rng.permutation(len(labeled))
            labeled = labeled[rand_idx_labeled]

        # get labels
        if which_set != 'unlabeled':
            data_y = data['labs_ex'][labeled_indices]
            data_y = data_y - 1

            data_y_identity = data['labs_id'][labeled_indices]

            self.one_hot = one_hot
            if one_hot:
                one_hot = np.zeros((data_y.shape[0], 7))
                for i in xrange(data_y.shape[0]):
                    one_hot[i, data_y[i]] = 1.
                data_y = one_hot.astype('float32')

            if shuffle:
                data_y = data_y[rand_idx_labeled]
                data_y_identity = data_y_identity[rand_idx_labeled]

        else:
            data_y = None
            data_y_identity = None

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size,
                                                                   image_size,
                                                                   1),
                                                                  axes)

        # init the super class
        super(TFDSemi, self).__init__(X=features, V=features, L=labeled,
                                  y=data_y, view_converter=view_converter)

        assert not contains_nan(self.X)
        assert not contains_nan(self.L)

        self.y_identity = data_y_identity
        self.axes = axes

        if preprocessor is not None:
            preprocessor.apply(self)

    def get_test_set(self, fold=None):
        """
        Return the test set
        """

        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        if fold is not None:
            args['fold'] = fold

        return TFDSemi(**args)
