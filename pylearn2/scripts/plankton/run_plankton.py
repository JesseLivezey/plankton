from pylearn2.config import yaml_parse
import numpy as np
import sys

with open(sys.argv[1], 'rb') as f:
    train = f.read()
    
init_alpha = .1
dim = 784

init_bias = 0.
L0dim = 500
L1dim = 500
L2dim = 500
zdim = 2
ydim = 10
L_3dim = 500
L_2dim = 500
L_1dim = 500
max_col_norm = .9

params = {'L0dim': L0dim,
          'L1dim': L1dim,
          'L2dim': L2dim,
          'zdim': zdim,
          'L_3dim': L_3dim,
          'L_2dim': L_2dim,
          'L_1dim': L_1dim,
          'L0std': np.sqrt(init_alpha/(dim+L0dim)),
          'L1std': np.sqrt(init_alpha/(L0dim+L1dim)),
          'L2std': np.sqrt(init_alpha/(L1dim+L2dim)),
          'zstd': np.sqrt(init_alpha/(L1dim+ydim+zdim)),
          'ystd': np.sqrt(init_alpha/(L1dim+ydim+zdim)),
          'L_3std': np.sqrt(init_alpha/(ydim+zdim+L_3dim)),
          'L_2std': np.sqrt(init_alpha/(L_3dim+L_2dim)),
          'L_1std': np.sqrt(init_alpha/(L_1dim+L_2dim)),
          'L_0std': np.sqrt(init_alpha/(dim+L_1dim)),
          'max_col_norm': max_col_norm,
          'init_bias': init_bias
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
