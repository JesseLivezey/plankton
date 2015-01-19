from pylearn2.blocks import Block
import numpy as np
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import Conv2DSpace
from scipy.ndimage.interpolation import rotate, shift

class DataAugmentation(Block):

    def __init__(self, space, seed=20150111, spline_order=1, cval=0.):
        self.rng = make_np_rng(np.random.RandomState(seed),
                               which_method=['rand', 'randint'])
        assert isinstance(space, Conv2DSpace)
        self.space = space
        self.spline_order = spline_order
        self.cval = cval
        super(DataAugmentation, self).__init__()

    def set_rand(self):
        self.p = self.rng.rand()
        self.deg = 360*self.rng.rand()
        self.axis0 = self.rng.uniform(low=-3., high=3.)
        self.axis1 = self.rng.uniform(low=-3., high=3.)

    def perform(self, X):
        X = np.transpose(X, axes=(1,2,3,0))
        if self.p >= .5:
            X = X[::-1]
        X = rotate(X, angle=self.deg, order=self.spline_order, cval=self.cval)
        X = shift(X, shift=(self.axis0, self.axis1, 0,0), cval=self.cval)
        X = np.transpose(X, axes=(3,0,1,2))
        return X

    def get_input_space(self):
        return self.space

    def get_output_space(self):
        return self.space

