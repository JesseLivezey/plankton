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
        labeled_act = model.encode(rval)[0]
        cost = model.labeled_layer.cost(Y, labeled_act)
        cost.name = 'label_xent'
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        targets = model.labeled_layer.get_output_space()
        return (CompositeSpace((features, targets)), ('labeled', 'labels'))

class TwoLabelXEnt(DefaultDataSpecsMixin, Cost):
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
        L, Y1, Y2 = data
        rval = L
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled1, labeled2, unlabeled = composite.layers
                labeled_act1 = labeled1.fprop(rval)
                labeled_act2 = labeled2.fprop(rval)
                cost = labeled1.cost(Y1, labeled_act1)+\
                       labeled2.cost(Y2, labeled_act2)
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets1 = layer.raw_layer.layers[0].get_output_space()
                targets2 = layer.raw_layer.layers[1].get_output_space()
                break
        return (CompositeSpace((features, targets1, targets2)), ('labeled', 'labels1', 'labels2'))

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

class LabelHingeL2(DefaultDataSpecsMixin, Cost):
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
                a = 1.-labeled_act*Y
                a *= a>0.
                cost = (0.5*T.sqr(a)).mean(0).sum()
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
                cost = .5*T.sqr(Y-labeled_act).mean(0).sum()
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
                rval = model.apply_dropout(
                        state=rval,
                        include_prob=include_prob,
                        theano_rng=theano_rng,
                        scale=scale,
                        mask_value=labeled.dropout_input_mask_value,
                        input_space=labeled.get_input_space(),
                        per_example=self.per_example
                        )
                labeled_act = labeled.fprop(rval)
                cost = .5*T.sqr(Y-labeled_act).mean(0).sum()
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

class LabelMisclass1(DefaultDataSpecsMixin, Cost):
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
        L, Y1 = data
        rval = L
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                sup1, sup2, unsup = composite.layers
                sup_act = sup1.fprop(rval)
                cost = T.neq(T.argmax(Y1, axis=1), T.argmax(sup_act, axis=1)).astype(theano.config.floatX).mean()
                break
            rval = layer.fprop(rval)
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[0].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels1'))

class LabelMisclass2(DefaultDataSpecsMixin, Cost):
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
        L, Y2 = data
        rval = L
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                sup1, sup2, unsup = composite.layers
                sup_act = sup2.fprop(rval)
                cost = T.neq(T.argmax(Y2, axis=1), T.argmax(sup_act, axis=1)).astype(theano.config.floatX).mean()
                break
            rval = layer.fprop(rval)
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets = layer.raw_layer.layers[1].get_output_space()
                break
        return (CompositeSpace((features, targets)), ('labeled', 'labels2'))

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

class TwoXCov(DefaultDataSpecsMixin, Cost):
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
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled1, labeled2, unlabeled = composite.layers
                labeled_act1 = labeled1.fprop(rval)
                labeled_act2 = labeled2.fprop(rval)
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act1 = labeled_act1-labeled_act1.mean(axis=0, keepdims=True)
                labeled_act2 = labeled_act2-labeled_act2.mean(axis=0, keepdims=True)
                unlabeled_act = unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc1 = T.dot(labeled_act1.T, unlabeled_act)/N
                cc2 = T.dot(labeled_act2.T, unlabeled_act)/N
                cost = .5*T.sqr(cc1).sum() + .5*T.sqr(cc2).sum() 
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')

class LabelTwoXCov(DefaultDataSpecsMixin, Cost):
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
        X,y1,y2 = data
        N = X.shape[0].astype(theano.config.floatX)
        rval = X
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled1, labeled2, unlabeled = composite.layers
                labeled_act1 = y1
                labeled_act2 = y2
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act1 = labeled_act1-labeled_act1.mean(axis=0, keepdims=True)
                labeled_act2 = labeled_act2-labeled_act2.mean(axis=0, keepdims=True)
                unlabeled_act = unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc1 = T.dot(labeled_act1.T, unlabeled_act)/N
                cc2 = T.dot(labeled_act2.T, unlabeled_act)/N
                cost = .5*T.sqr(cc1).sum() + .5*T.sqr(cc2).sum() 
                break
            rval = layer.fprop(rval)
        return cost

    def get_data_specs(self, model):
        features = model.get_input_space()
        for layer in model.layers:
            if isinstance(layer, FlattenerLayer):
                targets1 = layer.raw_layer.layers[0].get_output_space()
                targets2 = layer.raw_layer.layers[1].get_output_space()
                break
        return (CompositeSpace((features, targets1, targets2)), ('labeled', 'labels1', 'labels2'))

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
            if isinstance(layer, FlattenerLayer):
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

class AbsXCov(DefaultDataSpecsMixin, Cost):
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
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = labeled.fprop(rval)
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
                unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc = T.dot(labeled_act.T, unlabeled_act)/N
                cost = T.abs_(cc).sum()
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')

class XCor(DefaultDataSpecsMixin, Cost):
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
            if isinstance(layer, FlattenerLayer):
                composite = layer.raw_layer
                labeled, unlabeled = composite.layers
                labeled_act = labeled.fprop(rval)
                unlabeled_act = unlabeled.fprop(rval)

                labeled_act = labeled_act-labeled_act.mean(axis=0, keepdims=True)
                unlabeled_act =unlabeled_act-unlabeled_act.mean(axis=0, keepdims=True)
                cc = T.dot(labeled_act.T, unlabeled_act)/N
                corr = (1./T.sqrt(T.sqr(labeled_act).mean(axis=0, keepdims=True))).T*cc*(1./T.sqrt(T.sqr(unlabeled_act).mean(axis=0, keepdims=True)))
                cost = .5*T.sqr(corr).sum()
                break
            rval = layer.fprop(rval)
        return cost
    
    def get_data_specs(self, model):
        features = model.get_input_space()
        return (features, 'features')
