from pylearn2.blocks import Block
import numpy as np
from pylearn2.utils.rng import make_np_rng

class DataAugmentation(Block):

    def __init__(self, seed=20150111):
        self.rng = make_np_rng(np.random.RandomState(seed))
        super(DataAugmentation, self).__init__()

    def perform(self, X):
        if self.rng.rand() >= .5:
            X = X[:,::-1]
        X = np.transpose(X, axes=(1,2,3,0))
        k = self.rng.randint(4)
        X = np.rot90(X, k=k)
        X = np.transpose(X, axes=(3,0,1,2))
        return X

