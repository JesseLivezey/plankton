from pylearn2.models.mlp import Layer, MLP, FlattenerLayer
from pylearn2.utils import wraps
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

class EncodeDecode(MLP):

    def encode(self, state_below):
        rval = state_below
        for layer in self.layers:
            if isinstance(layer, FlattenerLayer):
                rvals = tuple([l.fprop(rval) for l in layer.raw_layer.layers])
                rval = rvals
                break
            rval = layer.fprop(rval)
        return rval

    def dropout_encode(self, state_below, default_input_include_prob=0.5,
            input_include_probs=None, default_input_scale=2.,
            input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                    state=state_below,
                    include_prob=include_prob,
                    theano_rng=theano_rng,
                    scale=scale,
                    mask_value=layer.dropout_input_mask_value,
                    input_space=layer.get_input_space(),
                    per_example=per_example
                    )
            if isinstance(layer, FlattenerLayer):
                rvals = tuple([l.fprop(state_below) for l in layer.raw_layer.layers])
                rval = rvals
            state_below = layer.fprop(state_below)

        return rval

    @property
    def labeled_layer(self):
        for layer in self.layers:
            if isinstance(layer, FlattenerLayer):
                labeled, unlabeled = layer.raw_layer.layers
                break
        return labeled

    @property
    def unlabeled_layer(self):
        for layer in self.layers:
            if isinstance(layer, FlattenerLayer):
                labeled, unlabeled = layer.raw_layer.layers
                break
        return unlabeled
