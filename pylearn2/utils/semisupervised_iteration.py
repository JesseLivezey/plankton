"""
Iterators providing indices for different kinds of iteration over
datasets.

Presets:

- random_slice: on each call to next, returns a slice of the dataset,
  chosen uniformly at random over contiguous slices.
  Samples with replacement, but still reports that
  container is empty after num_examples / batch_size calls
- random_uniform: on each call to next, returns a random subset of the
  dataset. Samples with replacement, but still reports that
  container is empty after num_examples / batch_size calls
"""
from __future__ import division
import functools
import inspect
import numpy as np

from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_izip, wraps
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng

# Make sure that the docstring uses restructured text list format.
# If you change the module-level docstring, please re-run
# pylearn2/doc/scripts/docgen.py and make sure sphinx doesn't issue any
# warnings for this file.
# This particular docstring was being frequently broken prior to the
# addition of this test.
# TODO: have nosetests run docgen.py in warning=error mode, remove
# tests for specific conditions


class SubsetIterator(object):
    """
    An iterator that returns slices or lists of indices into a dataset
    of a given fixed size.

    Parameters
    ----------
    dataset_size : list of ints
        The number of examples, total, in the dataset.
    batch_size : int, optional
        The (typical/maximum) number of examples per batch. Less
        may be returned in the very last batch if batch size
        does not evenly divide `dataset_size`.
    num_batches : int, optional
        The number of batches to return. Needn't be specified
        if `batch_size` is specified. If both `batch_size` and
        `num_batches` are specified then it must be true that
        `batch_size * num_batches <= dataset_size`.
    rng : `np.random.RandomState` or seed, optional
        A `np.random.RandomState` object or the seed to be
        used to create one. A deterministic default seed is
        used otherwise.
    """
    # This breaks the doc generation, so until we figure out why, not in the
    # docstring.
    #
    # Attributes
    # ----------
    # batch_size : int
    # num_batches : int
    # num_examples : int
    # uneven : bool
    # fancy : bool
    #     `True` if this iterator produces lists of indices,
    #     `False` if it produces slices.
    # stochastic : bool
    #     `True` if this iterator makes use of the random number
    #     generator, and will therefore produce different sequences
    #     depending on the RNG state. `False` otherwise.

    def __init__(self, dataset_size, batch_size=None,
                 num_batches=None, rng=None):
        raise NotImplementedError()

    def next(self):
        """
        Retrieves description of the next batch of examples.

        Returns
        -------
        next_batch : `slice` or list of int
            An object describing the indices in the dataset of
            a batch of data. Either a `slice` object or a list
            of integers specifying individual indices of
            examples.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self

    # Does this return subsets that need fancy indexing? (i.e. lists
    # of indices)
    fancy = False

    # Does this class make use of random number generators?
    stochastic = False

    # Does it ensure that every batch has the same size?
    uniform_batch_size = False

    @property
    def batch_size(self):
        """
        The (maximum) number of examples in each batch.

        Returns
        -------
        batch_size : int
            The (maximum) number of examples in each batch. This is
            either as specified via the constructor, or inferred from
            the dataset size and the number of batches requested.
        """
        return self._batch_size

    @property
    def num_batches(self):
        """
        The total number of batches that the iterator will ever return.

        Returns
        -------
        num_batches : int
            The total number of batches the iterator will ever return.
            This is either as specified via the constructor, or
            inferred from the dataset size and the batch size.
        """
        return self._num_batches

    @property
    def num_examples(self):
        """
        The total number of examples over which the iterator operates.

        Returns
        -------
        num_examples : int
            The total number of examples over which the iterator operates.
            May be less than the dataset size.
        """
        return self.batch_size * self.num_batches

    @property
    def uneven(self):
        """
        Whether every batch will be the same size.

        Returns
        -------
        uneven : bool
            `True` if returned batches may be of differing sizes,
            `False` otherwise.
        """
        raise NotImplementedError()

class RandomUniformSubsetIterator(SubsetIterator):
    """
    Selects minibatches of examples by drawing indices uniformly
    at random, with replacement.

    Notes
    -----
    Returns lists of indices (`fancy = True`).

    See :py:class:`SubsetIterator` for detailed constructor parameter
    and attribute documentation.
    """

    def __init__(self, dataset_size, tied_sets, sources, 
                 batch_size, num_batches, rng=None):
        self._rng = make_np_rng(rng, which_method=["random_integers",
                                                   "shuffle"])
        if batch_size is None:
            raise ValueError("batch_size cannot be None for random uniform "
                             "iteration")
        elif num_batches is None:
            raise ValueError("num_batches cannot be None for random uniform "
                             "iteration")
        for size in dataset_size:
            assert size >= batch_size
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0
        self._sources = sources
        assert len(dataset_size) == len(sources)
        for item_set in tied_sets:
            length = dataset_size[sources.index(item_set[0])]
            for item in item_set:
                assert length == dataset_size[sources.index(item)]
        self.set_size = {str(sets): dataset_size[sources.index(sets[0])] 
                         for sets in tied_sets}
        self.source_mapping = {}
        for source in sources:
            for sets in tied_sets:
                if source in sets:
                    self.source_mapping[source] = str(sets)

    @wraps(SubsetIterator.next)
    def next(self):
        if self._next_batch_no >= self._num_batches:
            raise StopIteration()
        else:
            set_indices = {}
            for sets in self.set_size.keys():
                set_indices[sets] = self._rng.random_integers(low=0,
                                       high=self.set_size[sets]-1,
                                       size=(self._batch_size,))
            source_indices = {}
            for source in self._sources:
                source_indices[source] = set_indices[self.source_mapping[source]]
            self._next_batch_no += 1
            return source_indices

    fancy = True
    stochastic = True
    uniform_batch_size = True


class RandomSliceSubsetIterator(RandomUniformSubsetIterator):
    """
    Returns minibatches that are randomly selected contiguous slices in
    index space.

    Notes
    -----
    Returns slice objects to represent ranges of indices (`fancy = False`).

    See :py:class:`SubsetIterator` for detailed constructor parameter
    and attribute documentation.
    """

    def __init__(self, dataset_size, tied_sets, sources, 
                 batch_size, num_batches, rng=None):
        if batch_size is None:
            raise ValueError("batch_size cannot be None for random slice "
                             "iteration")
        elif num_batches is None:
            raise ValueError("num_batches cannot be None for random slice "
                             "iteration")
        super(RandomSliceSubsetIterator, self).__init__(dataset_size,
                                                        tied_sets,
                                                        sources,
                                                        batch_size,
                                                        num_batches, rng)
        self._last_start = []
        for size in self._dataset_size:
            self._last_start.append(size-self._batch_size)
        for last_start in self._last_start:
            if self._last_start < 0:
                raise ValueError("batch_size > dataset_size not supported for "
                                 "random slice iteration")

    @wraps(SubsetIterator.next)
    def next(self):
        if self._next_batch_no >= self._num_batches:
            raise StopIteration()
        else:
            set_slices = {}
            for sets in self.set_size.keys():
                start = self._rng.random_integers(low=0, high=self.set_size[sets]-self._batch_size)
                set_slices[sets] = slice(start, start + self._batch_size)
            source_slices = {}
            for source in self._sources:
                source_slices[source] = set_slices[self.source_mapping[source]]
            self._next_batch_no += 1
            return source_slices

    fancy = False
    stochastic = True
    uniform_batch_size = True

_iteration_schemes = {
    'random_slice': RandomSliceSubsetIterator,
    'random_uniform': RandomUniformSubsetIterator,
}


def has_uniform_batch_size(mode):
    """
    Returns True if the iteration scheme has uniform batch size,
    False if not

    Parameters
    ----------
    mode: string
        A string defining an iteration scheme in _iteration_schemes

    Returns
    -------
    boolean
        True if the iteration scheme has uniform batch size,
        False otherwise
    """
    return resolve_iterator_class(mode).uniform_batch_size


def is_stochastic(mode):
    """

    """
    return resolve_iterator_class(mode).stochastic


def resolve_iterator_class(mode):
    """
    Map textual representations of default iteration modes to classes.

    Parameters
    ----------
    mode : str or class object
        If a string, identifier string for the built-in iteration modes.
        See the module documentation of :py:mod:`pylearn2.utils.iteration`
        for a list of available modes. If a class, it is expected to
        be a class that respects the constructor and attribute interface
        defined in :py:class:`SubsetIterator`.

    Returns
    -------
    subset_iter_class : class
        A class instance (i.e., an instance of type `type`) that
        interface defined in :py:class:`SubsetIterator`.
    """
    if isinstance(mode, basestring) and mode not in _iteration_schemes:
        raise ValueError("unknown iteration mode string: %s" % mode)
    elif mode in _iteration_schemes:
        subset_iter_class = _iteration_schemes[mode]
    else:
        subset_iter_class = mode
    return subset_iter_class


class FiniteDatasetIterator(object):
    """
    A wrapper around subset iterators that actually retrieves
    data.

    Parameters
    ----------
    dataset : `Dataset` object
        The dataset over which to iterate.
    data_specs : tuple
        A `(space, source)` tuple. See :ref:`data_specs` for a full
        description. Must not contain nested composite spaces.
    subset_iterator : object
        An iterator object that returns slice objects or lists of
        examples, conforming to the interface specified by
        :py:class:`SubsetIterator`.
    return_tuple : bool, optional
        Always return a tuple, even if there is exactly one source
        of data being returned. Defaults to `False`.
    convert : list of callables
        A list of callables, in the same order as the sources
        in `data_specs`, that will be called on the individual
        source batches prior to any further processing.

    Notes
    -----
    See the documentation for :py:class:`SubsetIterator` for
    attribute documentation.
    """

    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        # Keep only the needed sources in self._raw_data.
        # Remember what source they correspond to in self._source
        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        # the dataset's data spec is either a single (space, source) pair,
        # or a pair of (non-nested CompositeSpace, non-nested tuple).
        # We could build a mapping and call flatten(..., return_tuple=True)
        # but simply putting spaces, sources and data in tuples is simpler.
        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        all_data = self._dataset.get_data()
        if not isinstance(all_data, tuple):
            all_data = (all_data,)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._raw_data = tuple(all_data[dataset_source.index(s)]
                               for s in source)
        self._source = source

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

        self._sources = source
        for i, (so, sp, dt) in enumerate(safe_izip(source,
                                                   sub_spaces,
                                                   self._raw_data)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            if init_fn is None:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:

                    def fn(batch, dspace=dspace, sp=sp):
                        try:
                              return dspace.np_format_as(batch, sp)
                        except ValueError as e:
                            msg = str(e) + '\nMake sure that the model and '\
                                           'dataset have been initialized with '\
                                           'correct values.'
                            reraise_as(ValueError(msg))
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn

    def __iter__(self):
        return self

    @wraps(SubsetIterator.next)
    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using np.take()

        rval = tuple(
            fn(data[next_index[source]]) if fn else data[next_index[source]]
            for data, fn, source in safe_izip(self._raw_data, self._convert, self._source))
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic
