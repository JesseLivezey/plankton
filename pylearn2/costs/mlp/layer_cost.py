"""
Costs for use with individual layers in MLP.
"""
__authors__ = 'Jesse Livezey'

import theano
from theano import tensor as T

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as
from pylearn2.space import CompositeSpace
from pylearn2.models.mlp import FlattenerLayer


class FlattenerCost(DefaultDataSpecsMixin, Cost):
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
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = labeled.fprop(rval)
                cost = .5*T.sqr(Y-labeled_act).sum()
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

class MLPAE(DefaultDataSpecsMixin, Cost):
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
        cost = .5*(T.sqr(V-X_hat).sum())
        cost.name = 'ae_mse'
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (CompositeSpace((features, features)), ('features', 'targets'))

class FlattenerMisclass(DefaultDataSpecsMixin, Cost):
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
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                sup, unsup = composite.layers
                sup_act = sup.fprop(rval)
                cost = T.neq(T.argmax(Y, axis=1), T.argmax(sup_act, axis=1)).astype(theano.config.floatX).mean()
                break
            rval = layer.fprop(rval)
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

class XCov(DefaultDataSpecsMixin, Cost):
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
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        rval = X
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = labeled.fprop(rval)
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
                unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc = T.dot(labeled_act.T, unlabeled_act)/N
                cost = .5*T.sqr(cc).sum()
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')
