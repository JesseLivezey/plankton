from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import theano, os, cPickle
import theano.tensor as T
import numpy as np

import sys, csv

sharpen = 1.

model_path = sys.argv[1]
data_path = sys.argv[2]

with open(os.path.join(data_path,'label_mapping.pkl'), 'r') as f:
    label_mapping = cPickle.load(f)

model = serial.load(model_path)
train = model.dataset_yaml_src
ds = yaml_parse.load(train)
data = ds.raw.unlabeled-ds.raw.feature_mean
im_shape = data.shape[1:]
topo_view = data.reshape(-1, np.prod(im_shape))
ids = ds.raw.ids_unlabeled

n_examples = topo_view.shape[0]

X = model.get_input_space().make_theano_batch()

pred_symb = model.fprop(X)
pred = theano.function([X], pred_symb)

batch_size = 100
leftover = n_examples % batch_size
predictions = np.zeros((n_examples,121), dtype='float32')

for ii in xrange(int(n_examples/batch_size)):
    batch = data[ii*batch_size:(ii+1)*batch_size]
    batch = np.transpose(batch, axes=(1,2,3,0))
    for jj in xrange(2):
        for kk in xrange(4):
            batch_new = batch.copy()
            if jj == 1:
                batch_new = batch_new[::-1]
            batch_new = np.rot90(batch_new, k=kk)
            batch_new = np.transpose(batch_new, axes=(3,0,1,2))
            predictions[ii*batch_size:(ii+1)*batch_size] += pred(batch_new)

batch = data[-leftover:]
batch = np.transpose(batch, axes=(1,2,3,0))
for jj in xrange(2):
    for kk in xrange(4):
        batch_new = batch.copy()
        if jj == 1:
            batch_new = batch_new[::-1]
        batch_new = np.rot90(batch_new, k=kk)
        batch_new = np.transpose(batch_new, axes=(3,0,1,2))
        predictions[-leftover:] += pred(batch_new)
if sharpen != 1.:
    predictions = np.power(predictions, sharpen)

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




