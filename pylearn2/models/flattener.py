__authors__ = 'Jesse Livezey'

from pylearn2.utils import wraps
from pylearn2.models.mlp import FlattenerLayer, Layer

class FlattenerLayer2(FlattenerLayer):
    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        if targets is not None:
             targets=self.get_target_space().format_as(targets, self.raw_layer.get_target_space())
        return self.raw_layer.get_layer_monitoring_channels(
             state_below=state_below,
             state=self.get_output_space().format_as(state, self.raw_layer.get_output_space()),
             targets=targets
             )


