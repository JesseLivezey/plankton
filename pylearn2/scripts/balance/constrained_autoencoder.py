from pylearn2.config import yaml_parse
import numpy as np
with open('semi.yaml', 'rb') as f:
    train = f.read()
    
init_alpha = .1
dim = 2304

init_bias = 0.
L0dim = 400
L1dim = 400
L2dim = 400
zdim = 393
ydim=7
L_3dim=1000
L_2dim=1000
L_1dim=1000
max_col_norm = .863105108422

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
