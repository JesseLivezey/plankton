"""
Dataset for training with combination of supervised and unsupervised
data. Based on the DenseDesignMatrix class.
"""
__authors__ = "Jesse Livezey"

import functools

import logging
import warnings

import numpy as np

from pylearn2.datasets import cache
from pylearn2.utils.semisupervised_iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

import copy

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets import control
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from theano import config


logger = logging.getLogger(__name__)


class SemiSupervised(Dataset):
    """
    A class for representing datasets that can be stored as a dense design
    matrix of supervised and unsupervised data.


    Parameters
    ----------
    X : ndarray, 2-dimensional
        A design matrix\
        of shape (number examples, number features) \
        that defines the supervised dataset.
    V : ndarry, 2-dimensional
        A design matrix\
        of shape (number examples, number features) \
        that defines the targets of the network.
    L : ndarry, 2-dimensional
        A design matrix\
        of shape (number examples, number features) \
        that defines the supervised examples of the network.
    y : ndarray, optional
        Labels for each supervised example (e.g., class ids, values to be predicted
        in a regression task).

        Format should be:

        - 2D ndarray, data type optional:
            This is the most common format and can be used for a variety
            of problem types. Each row of the matrix becomes the target
            for a different example. Specific models / costs can interpret
            this target vector differently. For example, the `Linear`
            output layer for the `MLP` class expects the target for each
            example to be a vector of real-valued regression targets. (It
            can be a vector of size one if you only have one regression
            target). The `Softmax` output layer of the `MLP` class expects
            the target to be a vector of N elements, where N is the number
            of classes, and expects all but one of the elements to 0. One
            element should have value 1., and the index of this element
            identifies the target class.
    view_converter : object, optional
        An object for converting between the design matrix \
        stored internally and the topological view of the data.
    rng : object, optional
        A random number generator used for picking random \
        indices into the design matrix when choosing minibatches.
    X_labels : int, optional
        If X contains labels then X_labels must be passed to indicate the
        total number of possible labels e.g. the size of a the vocabulary
        when X contains word indices. This will make the set use
        IndexSpace. Also applied to V.
    y_labels : int, optional
        If y contains labels then y_labels must be passed to indicate the
        total number of possible labels e.g. 10 for the MNIST dataset
        where the targets are numbers. This will make the set use
        IndexSpace.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, X, V, L, y,
                 view_converter=None, axes=('b', 0, 1, 'c'),
                 rng=_default_seed, X_labels=None, y_labels=None):
        self.X = X
        self.V = V
        self.L = L
        self.y = y
        self.view_converter = view_converter
        self.X_labels = X_labels
        self.y_labels = y_labels
        self._tied_sets = [['features', 'targets'], ['labeled', 'labels']]
        self.conv_sources = ['features', 'targets', 'labeled']

        self._check_labels()
        
        assert X.shape[1] == L.shape[1], ("Labeled and unlabeled "
                                         "features have different shapes.")

        if view_converter is not None:
            # Get the topo_space (usually Conv2DSpace) from the
            # view_converter
            if not hasattr(view_converter, 'topo_space'):
                raise NotImplementedError("Not able to get a topo_space "
                                          "from this converter: %s"
                                          % view_converter)

            # self.X_topo_space stores a "default" topological space that
            # will be used only when self.iterator is called without a
            # data_specs, and with "topo=True", which is deprecated.
            self.X_topo_space = view_converter.topo_space
        else:
            self.X_topo_space = None

        # Update data specs, if not done in set_topological_view
        X_source = ('features', 'labeled')
        if X_labels is None:
            X_space = VectorSpace(dim=X.shape[1])
        else:
            if X.ndim == 1:
                dim = 1
            else:
                dim = X.shape[-1]
            X_space = IndexSpace(dim=dim, max_labels=X_labels)
        if y.ndim == 1:
            dim = 1
        else:
            dim = y.shape[-1]

        y_source = ('targets', 'labels')
        if y_labels is not None:
            y_space = IndexSpace(dim=dim, max_labels=y_labels)
        else:
            y_space = VectorSpace(dim=dim)

        space = CompositeSpace((X_space, X_space, X_space, y_space))
        source = X_source+y_source
        self.data_specs = (space, source)
        self.X_space = (X_space, X_space)

        self.compress = False
        self.design_loc = None
        self.rng = make_np_rng(rng, which_method="random_integers")
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('random_slice')
        self._iter_topo = False
        self._iter_targets = False
        self._iter_data_specs = (CompositeSpace(self.X_space), ('features', 'labeled'))

    def _check_labels(self):
        """Sanity checks for X_labels and y_labels."""
        if self.X_labels is not None:
            assert self.X is not None
            assert self.view_converter is None
            assert self.X.ndim <= 2
            assert self.V.ndim <= 2
            assert self.L.ndim <= 2
            assert np.all(self.V < self.X_labels)
            assert np.all(self.X < self.X_labels)
            assert np.all(self.L < self.X_labels)

        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            assert np.all(self.y < self.y_labels)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):

        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if src in self.conv_sources and \
               getattr(self, 'view_converter', None) is not None:
                conv_fn = (lambda batch, self=self, space=sp:
                           self.view_converter.get_formatted_batch(batch,
                                                                   space))
            else:
                conv_fn = None

            convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        dataset_size = [data.shape[0] for data in self.get_data()]
        tied_sets = self._tied_sets
        return FiniteDatasetIterator(self,
                                     mode(dataset_size,
                                          tied_sets,
                                          self.data_specs[1],
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def get_data(self):
        """
        Returns all the data, as it is internally stored.
        The definition and format of these data are described in
        `self.get_data_specs()`.

        Returns
        -------
        data : numpy matrix or 2-tuple of matrices
            The data
        """
        return (self.X, self.L, self.V, self.y)

    def get_topo_batch_axis(self):
        """
        The index of the axis of the batches

        Returns
        -------
        axis : int
            The axis of a topological view of this dataset that corresponds
            to indexing over different examples.
        """
        axis = self.view_converter.axes.index('b')
        return axis


    def get_stream_position(self):
        """
        If we view the dataset as providing a stream of random examples to
        read, the object returned uniquely identifies our current position in
        that stream.
        """
        return copy.copy(self.rng)

    def set_stream_position(self, pos):
        """
        .. todo::

            WRITEME properly

        Return to a state specified by an object returned from
        get_stream_position.

        Parameters
        ----------
        pos : object
            WRITEME
        """
        self.rng = copy.copy(pos)

    def restart_stream(self):
        """
        Return to the default initial state of the random example stream.
        """
        self.reset_RNG()

    def reset_RNG(self):
        """
        Restore the default seed of the rng used for choosing random
        examples.
        """

        if 'default_rng' not in dir(self):
            self.default_rng = make_np_rng(None, [17, 2, 946],
                                           which_method="random_integers")
        self.rng = copy.copy(self.default_rng)

    def apply_preprocessor(self, preprocessor, can_fit=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        preprocessor : object
            preprocessor object
        can_fit : bool, optional
            WRITEME
        """
        preprocessor.apply(self, can_fit)

    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
            This parameter is not named X because X is generally used to
            refer to the design matrix for the current problem. In this
            case we want to make it clear that `mat` need not be the design
            matrix defining the dataset.
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset "
                            "that has no view converter")
        if mat is None:
            mat = self.X
        return self.view_converter.design_mat_to_topo_view(mat)

    def get_formatted_view(self, mat, dspace):
        """
        Convert an array (or the entire dataset) to a destination space.

        Parameters
        ----------
        mat : ndarray, 2-dimensional
            An array containing a design matrix representation of
            training examples.

        dspace : Space
            A Space we want the data in mat to be formatted in.
            It can be a VectorSpace for a design matrix output,
            a Conv2DSpace for a topological output for instance.
            Valid values depend on the type of `self.view_converter`.

        Returns
        -------
        WRITEME
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_formatted_view on a dataset "
                            "that has no view converter")

        self.X_space.np_validate(mat)
        return self.view_converter.get_formatted_batch(mat, dspace)

    def get_weights_view(self, mat):
        """
        .. todo::

            WRITEME properly

        Return a view of mat in the topology preserving format. Currently
        the same as get_topological_view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional
            WRITEME
        """

        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        .. todo::

            Why is this parameter named 'V'?

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of
            training examples.
        axes : WRITEME
        """
        assert not contains_nan(V)
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels],
                                                   axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not contains_nan(self.X)

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            # This is to support old pickled models
            if getattr(self, 'y_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            elif getattr(self, 'max_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.max_labels)
            else:
                y_space = VectorSpace(dim=dim)
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

    def get_design_matrix(self, topo=None):
        """
        Return topo (a batch of examples in topology preserving format),
        in design matrix format.

        Parameters
        ----------
        topo : ndarray, optional
            An array containing a topological representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

        Returns
        -------
        WRITEME
        """
        if topo is not None:
            if self.view_converter is None:
                raise Exception("Tried to convert from topological_view to "
                                "design matrix using a dataset that has no "
                                "view converter")
            return self.view_converter.topo_view_to_design_mat(topo)

        return self.X

    def set_design_matrix(self, X):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : ndarray
            WRITEME
        """
        assert len(X.shape) == 2
        assert not contains_nan(X)
        self.X = X

    def get_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """

        warnings.warn("num_examples() is being deprecated, and will be "
                      "removed around November 7th, 2014. `get_num_examples` "
                      "should be used instead.",
                      stacklevel=2)

        return self.get_num_examples()

    def get_batch_design(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : int
            WRITEME
        include_labels : bool
            WRITEME
        """
        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                reraise_as(ValueError("Requested %d examples from a dataset "
                                      "containing only %d." %
                                      (batch_size, self.X.shape[0])))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        rx = np.cast[config.floatX](rx)
        return rx

    def get_batch_topo(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : int
            WRITEME
        include_labels : bool
            WRITEME
        """

        if include_labels:
            batch_design, labels = self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels

        return rval

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.X.shape[0]

    def view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.view_converter.view_shape()

    def weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.view_converter.weights_view_shape()

    def has_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y is not None

    def restrict(self, start, stop):
        """
        .. todo::

            WRITEME properly

        Restricts the dataset to include only the examples
        in range(start, stop). Ignored if both arguments are None.

        Parameters
        ----------
        start : int
            start index
        stop : int
            stop index
        """
        assert (start is None) == (stop is None)
        if start is None:
            return
        assert start >= 0
        assert stop > start
        assert stop <= self.X.shape[0]
        assert self.X.shape[0] == self.y.shape[0]
        self.X = self.X[start:stop, :]
        if self.y is not None:
            self.y = self.y[start:stop, :]
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == stop - start

    def convert_to_one_hot(self, min_class=0):
        """
        .. todo::

            WRITEME properly

        If y exists and is a vector of ints, converts it to a binary matrix
        Otherwise will raise some exception

        Parameters
        ----------
        min_class : int
            WRITEME
        """

        if self.y is None:
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix with no labels.")

        if self.y.ndim != 1:
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix whose labels aren't scalar.")

        if 'int' not in str(self.y.dtype):
            raise ValueError("Called convert_to_one_hot on a "
                             "DenseDesignMatrix whose labels aren't "
                             "integer-valued.")

        self.y = self.y - min_class

        if self.y.min() < 0:
            raise ValueError("We do not support negative classes. You can use "
                             "the min_class argument to remap negative "
                             "classes to positive values, but we require this "
                             "to be done explicitly so you are aware of the "
                             "remapping.")
        # Note: we don't check that the minimum occurring class is exactly 0,
        # since this dataset could be just a small subset of a larger dataset
        # and may not contain all the classes.

        num_classes = self.y.max() + 1

        y = np.zeros((self.y.shape[0], num_classes))

        for i in xrange(self.y.shape[0]):
            y[i, self.y[i]] = 1

        self.y = y

        # Update self.data_specs with the updated dimension of self.y
        init_space, source = self.data_specs
        X_space, init_y_space = init_space.components
        new_y_space = VectorSpace(dim=num_classes)
        new_space = CompositeSpace((X_space, new_y_space))
        self.data_specs = (new_space, source)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : ndarray
            The data to be adjusted
        """
        return X / np.abs(X).max()

    def adjust_to_be_viewed_with(self, X, ref, per_example=None):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        X : int
            WRITEME
        ref : float
            WRITEME
        per_example : obejct, optional
            WRITEME
        """
        if per_example is not None:
            logger.warning("ignoring per_example")
        return np.clip(X / np.abs(ref).max(), -1., 1.)

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def set_view_converter_axes(self, axes):
        """
        .. todo::

            WRITEME properly

        Change the axes of the view_converter, if any.

        This function is only useful if you intend to call self.iterator
        without data_specs, and with "topo=True", which is deprecated.

        Parameters
        ----------
        axes : WRITEME
            WRITEME
        """
        assert self.view_converter is not None

        self.view_converter.set_axes(axes)
        # Update self.X_topo_space, which stores the "default"
        # topological space, which is the topological output space
        # of the view_converter
        self.X_topo_space = self.view_converter.topo_space

