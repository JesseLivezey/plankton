"""
Costs for use with individual layers in MLP.
"""
__authors__ = 'Jesse Livezey'

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as
from pylearn2.space import CompositeSpace
from pylearn2.models.mlp import CompositeLayer


class LabelXEnt(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        rval = L
        labeled_act = model.encode(rval)[0]
        cost = model.labeled_layer.cost(Y, labeled_act)
        cost.name = 'label_xent'
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        targets = model.labeled_layer.get_output_space()
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))


class DropoutLabelXEnt(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        rval = L
        labeled_act = model.dropout_encode(rval,
                default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs,
                default_input_scale=self.default_input_scale,
                input_scales=self.input_scales,
                per_example=self.per_example)[0]
        cost = model.labeled_layer.cost(Y, labeled_act)
        cost.name = 'dropout_label_xent'
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        targets = model.labeled_layer.get_output_space()
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

    def get_monitoring_channels(self, model, data):
        space, source = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        ipt = L
        labeled_act = model.encode(ipt)[0]
        cost = model.labeled_layer.cost(Y, labeled_act)
        rval = OrderedDict()
        rval['nll'] = cost
        return rval


class AEMSE(DefaultDataSpecsMixin, Cost):
    """
    """
    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X, V = data
        X_hat = model.fprop(X)
        cost = .5*T.sqr(V-X_hat).mean(0).sum()
        cost.name = 'ae_mse'
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (CompositeSpace((features, features)), ('features', 'targets'))

class DropoutAEMSE(DefaultDataSpecsMixin, Cost):
    """
    """
    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X, V = data
        X_hat = model.dropout_fprop(X,
                default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs,
                default_input_scale=self.default_input_scale,
                input_scales=self.input_scales,
                per_example=self.per_example)
        cost = .5*T.sqr(V-X_hat).mean(0).sum()
        cost.name = 'dropout_ae_mse'
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (CompositeSpace((features, features)), ('features', 'targets'))

    def get_monitoring_channels(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X, V = data
        X_hat = model.fprop(X)
        cost = .5*T.sqr(V-X_hat).mean(0).sum()
        rval = OrderedDict()
        rval['ae_mse'] = cost
        return rval

class LabelMisclass(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        rval = L
        labeled_act = model.encode(rval)[0]
        cost = T.neq(T.argmax(Y, axis=1), T.argmax(labeled_act, axis=1)).astype(theano.config.floatX).mean()
        cost.name = 'misclass'
        return cost
    

    def get_data_specs(self, model):
        features = model.get_input_space()
        targets = model.labeled_layer.get_output_space()
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))


class XCov(DefaultDataSpecsMixin, Cost):
    """
    """

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        labeled_act, unlabeled_act = model.encode(X)
        labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
        unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
        cc = T.dot(labeled_act.T, unlabeled_act)/N
        cost = .5*T.sqr(cc).sum()
        cost.name = 'xcov'
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')


class N01Penalty(DefaultDataSpecsMixin, Cost):
    """
    """
    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        rval = X
        for layer in model.layers:
            if isinstance(layer, CompositeLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                unlabeled_act = unlabeled.fprop(rval)
                unlabeled_var = unlabeled_act.var(axis=0, keepdims=True)
                cost = -.5*(T.log(unlabeled_var) - T.sqr(unlabeled_act.mean(axis=0, keepdims=True)) - unlabeled_var).sum()
                break
            rval = layer.fprop(rval)
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')

class DropoutXCov(DefaultDataSpecsMixin, Cost):
    """
    """

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        labeled_act, unlabeled_act = model.dropout_encode(X)
        labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
        unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
        cc = T.dot(labeled_act.T, unlabeled_act)/N
        cost = .5*T.sqr(cc).sum()
        cost.name = 'dropout_xcov'
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')

    def get_monitoring_channels(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        labeled_act, unlabeled_act = model.dropout_encode(X)
        labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
        unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
        cc = T.dot(labeled_act.T, unlabeled_act)/N
        cost = .5*T.sqr(cc).sum()
        rval = OrderedDict()
        rval['xcov'] = cost
        return rval
