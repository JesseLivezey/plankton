from pylearn2.blocks import Block
import numpy as np
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import Conv2DSpace
from scipy.ndimage.interpolation import rotate

class DataAugmentation(Block):

    def __init__(self, space, seed=20150111):
        self.rng = make_np_rng(np.random.RandomState(seed), which_method=['rand', 'randint'])
        assert isinstance(space, Conv2DSpace)
        self.space = space
        super(DataAugmentation, self).__init__()

    def perform(self, X):
        X = np.transpose(X, axes=(1,2,3,0))
        if self.rng.rand() >= .5:
            X = X[::-1]
        k = 360.*self.rng.rand()
        X = rotate(X, angle=k, reshape=False)
        axis0 = self.rng.randint(low=-3, high=4)
        axis1 = self.rng.randint(low=-3, high=4)
        X = np.roll(X, shift=axis0, axis=0)
        X = np.roll(X, shift=axis1, axis=1)
        X = np.transpose(X, axes=(3,0,1,2))
        return X

    def get_input_space(self):
        return self.space

    def get_output_space(self):
        return self.space

