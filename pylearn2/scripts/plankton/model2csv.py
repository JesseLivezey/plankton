from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import theano, os, cPickle
import theano.tensor as T
import numpy as np

import sys, csv

model_path = sys.argv[1]
data_path = sys.argv[2]

with open(os.path.join(data_path,'label_mapping.pkl'), 'r') as f:
    label_mapping = cPickle.load(f)

model = serial.load(model_path)
train = model.dataset_yaml_src
ds = yaml_parse.load(train)
topo_view = ds.unlabeled-ds.feature_mean
data = topo_view.reshape(-1, 96*96)
ids = ds.ids_unlabeled

n_examples = topo_view.shape[0]

X = model.get_input_space().make_theano_batch()

pred_symb = model.fprop(X)
pred = theano.function([X], pred_symb)

batch_size = 100
leftover = n_examples % batch_size
predictions = np.zeros((n_examples,121), dtype='float32')

for ii in xrange(int(n_examples/batch_size)):
    batch = data[ii*batch_size:(ii+1)*batch_size]
    predictions[ii*batch_size:(ii+1)*batch_size] = pred(batch)

batch = data[-leftover:]
predictions[-leftover:] = pred(batch)

with open(os.path.join(data_path,'sampleSubmission.csv'), 'r') as f:
    reader = csv.reader(f)
    header = reader.next()

header_classes = header[1:]
pred_order = sorted(label_mapping.keys())
perm = [pred_order.index(cl) for cl in header_classes]
predictions = predictions[:,perm]
with open('submission.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for iid,pred in zip(ids,predictions):
        writer.writerow([str(iid)+'.jpg']+list(pred))




