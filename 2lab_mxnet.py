import os, numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
from mxnet.ndarray import load
import tensorflow as tf

from mxnet.gluon.data import Dataset, DataLoader
from mxnet import image

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import cv2

import time

#os.environ["MXNET_ENGINE_TYPE"] = 'NaiveEngine'
#os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'

def onehot(length, num):
    ans = np.zeros((length,))
    ans[num] = 1.
    return ans

class MyDataSet(Dataset):
    def __init__(self, root):
        self.img_paths = []
        for i in os.listdir(root):
            self.img_paths += [i+'/'+j for j in os.listdir(os.path.join(root,i))]
        self.img_paths = np.array(self.img_paths)
        np.random.shuffle(self.img_paths)
        self.img_paths = list(self.img_paths)
        self._img = os.path.join(root, '{}')
        self.labels = {i:onehot(len(os.listdir(root)), i0)
                  for i0,i in enumerate(os.listdir(root))}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self._img.format(self.img_paths[idx])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img[:,:,-1:]
        #print img.shape
        img = cv2.resize(img, (100,100), interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img * 1. / 255.
        img = img.ravel()
        #img = np.transpose(img, [2,0,1])
        img = nd.array(img.astype(np.float32), ctx=mx.gpu(0))

        lbl = nd.array(self.labels[img_path.split('/')[-2]], ctx=mx.gpu(0))

        return img, lbl

batch_size = 40
input_shape = (batch_size, 10000)

data = mx.sym.var('data', shape = input_shape)

def net0(data=data):
    stage1 = mx.sym.Convolution(data=data, name='stage1', num_filter=48, kernel=(7,7), stride=(2,2),pad=(1,1))
    stage2 = mx.sym.Pooling(data=stage1, kernel=(4,4), name='stage2',pool_type='max')
    stage3 = mx.sym.Convolution(data=stage2, name='stage3', num_filter=48, kernel=(4,4), pad=(1,1))
    stage4 = mx.sym.Convolution(data=stage3, name='stage4', num_filter=48, kernel=(4,4), pad=(1,1))
    stage5 = mx.sym.Pooling(data=stage4, kernel=(3,3), name='stage5',pool_type='max')
    stage6 = mx.sym.Convolution(data=stage5, name='stage6', num_filter=48, kernel=(4,4), pad=(1,1))
    stage7 = mx.sym.Convolution(data=stage6, name='stage7', num_filter=48, kernel=(4,4), pad=(1,1))
    stage8 = mx.sym.Pooling(data=stage7, kernel=(3, 3), name='stage8',pool_type='max')
    stage9 = mx.sym.flatten(stage8, name='flatten')

    stage10 = mx.sym.FullyConnected(data=stage9, num_hidden=300, name='stage10')
    stage11 = mx.sym.FullyConnected(data=stage10, num_hidden=300, name='stage11')
    out = mx.sym.FullyConnected(data=stage11, num_hidden=len(os.listdir('/home/alex/Desktop/labs/Cyrillic/')),
                                name='out')
    return out


#data = mx.sym.var('data', shape = input_shape)
out = mx.sym.softmax(net0(data=data),axis=1)

w=mx.viz.plot_network(out)
w.view('lab2')

net = mx.gluon.SymbolBlock(outputs=out, inputs=data)

ctx = mx.gpu(0)
net.collect_params().initialize(mx.init.Normal(), mx.gpu(0))

net.load_params('model1-0000.params', ctx=mx.gpu(0),
            allow_missing=True, ignore_extra=True)

mx.nd.waitall()
time.sleep(15)

PATH = '/home/alex/Desktop/labs/Cyrillic/'
my_train = MyDataSet(PATH)
train_loader = DataLoader(my_train, batch_size=batch_size, shuffle=False, last_batch='rollover')

num_steps = len(my_train)
trainer = gluon.Trainer(net.collect_params(), 'adam')#, {'learning_rate':0.0001})
criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=False)

class Accuracy(mx.metric.Accuracy):
    def __init__(self, logdir, tag, axis = 1, name='Accuracy'):
        super(Accuracy, self).__init__(axis = axis, name = name)
        self.writer = tf.summary.FileWriter(logdir)
        self.tag = tag
        self.iter = 0

    def update(self, labels, preds):

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis).asnumpy().astype('int32')
            label = mx.ndarray.argmax(label, axis=self.axis).asnumpy().astype('int32')
            #print pred_label.shape, label.shape
            self.sum_metric += np.sum(pred_label.ravel() == label.ravel())
            self.num_inst += len(pred_label.ravel())

        self.iter += 1

    def write_results(self, step, tag):

        acc = self.get()
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value = acc[1])#[1])
        self.writer.add_summary(summary, self.iter * step)

#PATH1m = '/'.join(PATH.split('/')[:-1])+'/acc'
#PATH2m = '/'.join(PATH.split('/')[:-1])+'/loss'
metrics = [Accuracy(logdir=PATH,tag='tag')]
#loss_metrics =
#valid_metrics = [Accuracy(logdir=PATH,tag='valid')]

num_epochs = 40
for epoch in range(num_epochs):
    t0 = time.time()
    total_loss = 0
    for m in metrics:
        m.reset()
    #for m in valid_metrics:
    #    m.reset()
    times = []
    for i0, (data, label) in enumerate(train_loader):
        # print data
        if i0 < 300:
            batch_size = 40  # data.shape[0]
            dlist = gluon.utils.split_and_load(data, [mx.gpu(0)])
            llist = gluon.utils.split_and_load(label, [mx.gpu(0)])

            with ag.record():

                t = time.time()
                preds = [net(X) for X in dlist]
                losses = []
                for i in range(len(preds)):
                    l = criterion(preds[i], llist[i])
                    losses.append(l)

            print 'batch', i0, 'loss', nd.mean(nd.array([nd.mean(losses[k]).asnumpy() for k in range(len(losses))],ctx=ctx)), 'epoch', epoch

            for l in losses:
                l.backward()
            total_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)

            for y in metrics:
                y.update(labels=llist, preds=preds)
                y.write_results(1,'train')#+i0)
        else:
            #for m in metrics:
            #    m.reset()

            print '========== Validating =========='
            batch_size = 40  # data.shape[0]
            dlist = gluon.utils.split_and_load(data, [mx.gpu(0)])
            llist = gluon.utils.split_and_load(label, [mx.gpu(0)])
            t = time.time()
            preds = [net(X) for X in dlist]
            losses = []
            for i in range(len(preds)):
                l = criterion(preds[i], llist[i])
                losses.append(l)
            print 'batch', i0, 'loss', nd.mean(nd.array([nd.mean(losses[k]).asnumpy() for k in range(len(losses))], ctx=ctx)), 'epoch', epoch

            for m in metrics:
                m.update(labels=llist, preds=preds)
                m.write_results(1, 'valid')#+i0-300)

    #for m in metrics:
    #    name, value = m.get()

    t1 = time.time()
    net.export('/home/alex/Desktop/labs/model1')
    print(epoch, t1 - t0, total_loss)#, name, value)

#net.export(PATH + 'model1')

