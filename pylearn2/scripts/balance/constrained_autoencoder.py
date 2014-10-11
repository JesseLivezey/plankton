from pylearn2.config import yaml_parse
import numpy as np
with open('semi2.yaml', 'rb') as f:
    train = f.read()
    
init_alpha = .1
dim = 2304

L0dim = 2000
L1dim = 2000
zdim = 793
ydim=7
L_2dim=2000
L_1dim=2000
max_col_norm = .863105108422

params = {'L0dim': L0dim,
          'L1dim': L1dim,
          'zdim': zdim,
          'L_2dim': L_2dim,
          'L_1dim': L_1dim,
          'L0std': np.sqrt(init_alpha/(dim+L0dim)),
          'L1std': np.sqrt(init_alpha/(L0dim+L1dim)),
          'zstd': np.sqrt(init_alpha/(L1dim+ydim+zdim)),
          'ystd': np.sqrt(init_alpha/(L1dim+ydim+zdim)),
          'L_2std': np.sqrt(init_alpha/(ydim+zdim+L_2dim)),
          'L_1std': np.sqrt(init_alpha/(L_1dim+L_2dim)),
          'L_0std': np.sqrt(init_alpha/(dim+L_1dim)),
          'max_col_norm': max_col_norm
          }
train = train % params
print train
train = yaml_parse.load(train)
train.main_loop()
