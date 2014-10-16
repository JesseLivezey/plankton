"""
Costs for use with individual layers in MLP.
"""
__authors__ = 'Jesse Livezey'

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet import binary_crossentropy

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as
from pylearn2.space import CompositeSpace
from pylearn2.models.mlp import FlattenerLayer


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
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = labeled.fprop(rval)
                cost = binary_crossentropy(labeled_act, Y).sum()
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

class DropoutLabelXEnt(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

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
        model._validate_layer_names(list(self.input_include_probs.keys()))
        model._validate_layer_names(list(self.input_scales.keys()))
        theano_rng = MRG_RandomStreams(max(model.rng.randint(2 ** 15), 1))

        space, sources = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        rval = L
        for layer in model.layers:
            layer_name = layer.layer_name
            if layer_name in self.input_include_probs:
                include_prob = self.input_include_probe[layer_name]
            else:
                include_prob = self.default_input_include_prob

            if layer_name in self.input_scales:
                scale = self.input_scales[layer_name]
            else:
                scale = self.default_input_scale

            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = model.apply_dropout(
                        state=rval,
                        include_prob=include_prob,
                        theano_rng=theano_rng,
                        scale=scale,
                        mask_value=labeled.dropout_input_mask_value,
                        input_space=labeled.get_input_space(),
                        per_example=self.per_example
                        )
                labeled_act = labeled.fprop(rval)
                cost = binary_crossentropy(labeled_act, Y).sum()
                break
            rval = model.apply_dropout(
                    state=rval,
                    include_prob=include_prob,
                    theano_rng=theano_rng,
                    scale=scale,
                    mask_value=layer.dropout_input_mask_value,
                    input_space=layer.get_input_space(),
                    per_example=self.per_example
                    )
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

class LabelMSE(DefaultDataSpecsMixin, Cost):
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

class DropoutLabelMSE(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

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
        model._validate_layer_names(list(self.input_include_probs.keys()))
        model._validate_layer_names(list(self.input_scales.keys()))
        theano_rng = MRG_RandomStreams(max(model.rng.randint(2 ** 15), 1))

        space, sources = self.get_data_specs(model)
        space.validate(data)
        L, Y = data
        rval = L
        for layer in model.layers:
            layer_name = layer.layer_name
            if layer_name in self.input_include_probs:
                include_prob = self.input_include_probe[layer_name]
            else:
                include_prob = self.default_input_include_prob

            if layer_name in self.input_scales:
                scale = self.input_scales[layer_name]
            else:
                scale = self.default_input_scale

            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = model.apply_dropout(
                        state=rval,
                        include_prob=include_prob,
                        theano_rng=theano_rng,
                        scale=scale,
                        mask_value=labeled.dropout_input_mask_value,
                        input_space=labeled.get_input_space(),
                        per_example=self.per_example
                        )
                labeled_act = labeled.fprop(rval)
                cost = .5*T.sqr(Y-labeled_act).sum()
                break
            rval = model.apply_dropout(
                    state=rval,
                    include_prob=include_prob,
                    theano_rng=theano_rng,
                    scale=scale,
                    mask_value=layer.dropout_input_mask_value,
                    input_space=layer.get_input_space(),
                    per_example=self.per_example
                    )
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

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
        cost = .5*(T.sqr(V-X_hat).sum())
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
        X_hat = model.dropout_fprop(X, self.default_input_include_prob,
                self.input_include_probs, self.default_input_scale,
                self.input_scales, self.per_example)
        cost = .5*(T.sqr(V-X_hat).sum())
        cost.name = 'dropout_ae_mse'
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        return (CompositeSpace((features, features)), ('features', 'targets'))

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
    
class DropoutXCov(DefaultDataSpecsMixin, Cost):
    """
    """
    supervised = True

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
        model._validate_layer_names(list(self.input_include_probs.keys()))
        model._validate_layer_names(list(self.input_scales.keys()))
        theano_rng = MRG_RandomStreams(max(model.rng.randint(2 ** 15), 1))

        space, sources = self.get_data_specs(model)
        space.validate(data)
        X = data
        N = X.shape[0].astype(theano.config.floatX)
        rval = X
        for layer in model.layers:
            layer_name = layer.layer_name
            if layer_name in self.input_include_probs:
                include_prob = self.input_include_probe[layer_name]
            else:
                include_prob = self.default_input_include_prob

            if layer_name in self.input_scales:
                scale = self.input_scales[layer_name]
            else:
                scale = self.default_input_scale
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = model.apply_dropout(
                        state=rval,
                        include_prob=include_prob,
                        theano_rng=theano_rng,
                        scale=scale,
                        mask_value=labeled.dropout_input_mask_value,
                        input_space=labeled.get_input_space(),
                        per_example=self.per_example
                        )
                labeled_act = labeled.fprop(rval)
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
                unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc = T.dot(labeled_act.T, unlabeled_act)/N
                cost = .5*T.sqr(cc).sum()
                break
            rval = model.apply_dropout(
                    state=rval,
                    include_prob=include_prob,
                    theano_rng=theano_rng,
                    scale=scale,
                    mask_value=layer.dropout_input_mask_value,
                    input_space=layer.get_input_space(),
                    per_example=self.per_example
                    )
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')
