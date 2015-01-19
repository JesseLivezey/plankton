from pylearn2.config import yaml_parse
import numpy as np
import sys

with open(sys.argv[1], 'rb') as f:
	# opens as string
    train = f.read()
    
# turns string into python object
train = yaml_parse.load(train)

train.main_loop()
