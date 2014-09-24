"""
Costs for use with individual layers in MLP.
"""
__authors__ = 'Jesse Livezey'

import theano
from theano import tensor as T

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as
from pylearn2.models.mlp import FlattenerLayer
from pylearn2.space import CompositeSpace


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
        X, Y = data
        rval = X
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                sup, unsup = composite.layers
                sup_act = sup.fprop(rval)
                mask = T.switch(T.lt(Y, 0),0,1)
                Y_mix = Y*mask+sup_act*(1-mask)
                cost = .5*T.sqr(Y_mix-sup_act).mean()
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('features', 'targets'))

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
        X = data
        rval = X
        X_hat = model.fprop(X)
        cost = .5*T.sqr(X-X_hat).mean()
        cost.name = 'ae_mse'
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')

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
        X, Y = data
        rval = X
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                sup, unsup = composite.layers
                sup_act = sup.fprop(rval)
                mask = T.switch(T.lt(Y, 0),0,1)
                Y_mix = Y*mask+sup_act*(1-mask)
                n_sup = (mask.mean(1)).sum().astype(theano.config.floatX)
                incorrect = T.neq(T.argmax(Y_mix, axis=1), T.argmax(sup_act, axis=1)).astype(theano.config.floatX).sum().astype(theano.config.floatX)
                cost = incorrect/n_sup
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('features', 'targets'))
