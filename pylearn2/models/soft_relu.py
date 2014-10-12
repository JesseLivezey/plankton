"""
Soft rectification layers.
"""
__authors__ = "Jesse Livezey"

import logging
import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.monitor import get_monitor_doc
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.utils import function
from pylearn2.utils import is_iterable
from pylearn2.utils import py_float_types
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils import contains_nan
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.models.mlp import Layer, Linear, ConvNonlinearity, ConvElemwise

from pylearn2.expr.nnet import (elemwise_kl, kl, compute_precision,
                                    compute_recall, compute_f1)

# Only to be used by the deprecation warning wrapper functions
from pylearn2.costs.mlp import L1WeightDecay as _L1WD
from pylearn2.costs.mlp import WeightDecay as _WD


logger = logging.getLogger(__name__)


class SoftRectifiedLinear(Linear):
    """
    Soft Rectified linear MLP layer.

    Parameters
    ----------
    elbow : float
        Parameter that controls softness of rectification.
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor.
    """

    def __init__(self, elbow=1.0, **kwargs):
        super(SoftRectifiedLinear, self).__init__(**kwargs)
        assert elbow > 0.
        self.elbow = elbow

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = (p+T.sqrt(self.elbow+T.sqr(p)))/2.
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()

class SoftRectifierConvNonlinearity(ConvNonlinearity):
    """
    A simple rectifier nonlinearity class for convolutional layers.

    Parameters
    ----------
    left_slope : float
        The slope of the left half of the activation function.
    """
    def __init__(self, elbow=1.0):
        """
        Parameters
        ----------
        elbow : float, optional
            Parameter determines softness of rectification.
            default is 0.0.
        """
        self.non_lin_name = "rectifier"
        self.elbow = elbow

    @wraps(ConvNonlinearity.apply)
    def apply(self, linear_response):
        """
        Applies the rectifier nonlinearity over the convolutional layer.
        """
        p = (p+T.sqrt(self.elbow+T.sqr(p)))/2.
        return p


class ConvSoftRectifiedLinear(ConvElemwise):
    """
    A convolutional rectified linear layer, based on theano's B01C
    formatted convolution.

    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    irange : float
        if specified, initializes each weight randomly in
        U(-irange, irange)
    border_mode : str
        A string indicating the size of the output:

        - "full" : The output is the full discrete linear convolution of the
            inputs.
        - "valid" : The output consists only of those elements that do not
            rely on the zero-padding. (Default)

    include_prob : float
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 0.
    init_bias : float
        All biases are initialized to this number
    W_lr_scale : float
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    left_slope : float
        The slope of the left half of the activation function
    max_kernel_norm : float
        If specifed, each kernel is constrained to have at most this norm.
    pool_type :
        The type of the pooling operation performed the the convolution.
        Default pooling type is max-pooling.
    tied_b : bool
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently.
    detector_normalization : callable
        See `output_normalization`
    output_normalization : callable
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

        - detector: the rectifier units can be normalized prior to the
            spatial pooling
        - output: the output of the layer, after spatial pooling, can
            be normalized as well

    kernel_stride : tuple
        The stride of the convolution kernel. A two-tuple of ints.
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 elbow=1.0,
                 max_kernel_norm=None,
                 pool_type='max',
                 tied_b=False,
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1),
                 monitor_style="classification"):

        nonlinearity = SoftRectifierConvNonlinearity(elbow)

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear and not both.")

        #Alias the variables for pep8
        mkn = max_kernel_norm
        dn = detector_normalization
        on = output_normalization

        super(ConvSoftRectifiedLinear, self).__init__(output_channels,
                                                  kernel_shape,
                                                  layer_name,
                                                  nonlinearity,
                                                  irange=irange,
                                                  border_mode=border_mode,
                                                  sparse_init=sparse_init,
                                                  include_prob=include_prob,
                                                  init_bias=init_bias,
                                                  W_lr_scale=W_lr_scale,
                                                  b_lr_scale=b_lr_scale,
                                                  pool_shape=pool_shape,
                                                  pool_stride=pool_stride,
                                                  max_kernel_norm=mkn,
                                                  pool_type=pool_type,
                                                  tied_b=tied_b,
                                                  detector_normalization=dn,
                                                  output_normalization=on,
                                                  kernel_stride=kernel_stride,
                                                  monitor_style=monitor_style)


