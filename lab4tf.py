#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt, os, json, cv2
import tensorflow as tf

#session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
# please do not use the totality of the GPU memory
#session_config.gpu_options.per_process_gpu_memory_fraction = 0.90


img_paths = []
for i0,i in enumerate(os.listdir('Cyrillic')):
    img_paths += ['Cyrillic'+'/'+i+'/'+j for j in os.listdir('Cyrillic/'+i)]
        
img_paths = np.array(img_paths)
np.random.shuffle(img_paths)
np.random.shuffle(img_paths)
img_paths = list(img_paths)
names_dict = sorted(list(np.unique([i.split('/')[1] for i in img_paths])))
names_dict = {names_dict[i]:i for i in range(len(names_dict))}

def onehot(length, num):
    ans = np.zeros((length,))
    ans[num] = 1.
    return ans

tf.reset_default_graph()
X0 = tf.placeholder(dtype=tf.float32, shape=[None,10000])
#Y0 = tf.placeholder(dtype=tf.float32, shape = [None,len(os.listdir('Cyrillic'))])

""" Net's architecture """
def net0(X0):
    
    stage1 = tf.contrib.layers.fully_connected(X0, num_outputs=700, activation_fn=None)
    stage2 = tf.contrib.layers.fully_connected(stage1, num_outputs=500, activation_fn=None)
    stage3 = tf.contrib.layers.fully_connected(stage2, num_outputs=300, activation_fn=None)
    stage4 = tf.contrib.layers.fully_connected(stage3, num_outputs=300, activation_fn=None)
    stage5 = tf.contrib.layers.fully_connected(stage4, num_outputs=200, activation_fn=None)
    stage6 = tf.contrib.layers.fully_connected(stage5, num_outputs=200, activation_fn=None)
    out = tf.contrib.layers.fully_connected(stage6, num_outputs=len(os.listdir('Cyrillic')), \
                                            activation_fn=None)    
    """
    stage1 = mx.sym.FullyConnected(data=data, num_hidden=700, name='stage1')
    stage2 = mx.sym.FullyConnected(data=stage1, num_hidden=500, name='stage2')
    stage3 = mx.sym.FullyConnected(data=stage2, num_hidden=300, name='stage3')
    stage4 = mx.sym.FullyConnected(data=stage3, num_hidden=300, name='stage4')
    stage5 = mx.sym.FullyConnected(data=stage4, num_hidden=200, name='stage5')
    stage6 = mx.sym.FullyConnected(data=stage5, num_hidden=100, name='stage6')
    out = mx.sym.FullyConnected(data=stage6, num_hidden=len(os.listdir('/home/alex/Desktop/labs/Cyrillic/')),
                                name='out')
    """
    return out

def net1(X1):
    out1 = tf.contrib.layers.fully_connected(X1, num_outputs=len(os.listdir('Cyrillic')), \
                                            activation_fn=None)
    stage61 = tf.contrib.layers.fully_connected(out1, num_outputs=200, activation_fn=None)
    stage51 = tf.contrib.layers.fully_connected(stage61, num_outputs=200, activation_fn=None)
    stage41 = tf.contrib.layers.fully_connected(stage51, num_outputs=300, activation_fn=None)
    stage31 = tf.contrib.layers.fully_connected(stage41, num_outputs=300, activation_fn=None)
    stage21 = tf.contrib.layers.fully_connected(stage31, num_outputs=500, activation_fn=None)
    stage11 = tf.contrib.layers.fully_connected(stage21, num_outputs=700, activation_fn=None)
    stageout = tf.contrib.layers.fully_connected(stage11, num_outputs=10000, activation_fn=None)
    return stageout
    
    
X = net0(X0)
Xout = net1(X)
#X = tf.clip_by_value(X, -10, 10)

vars = tf.trainable_variables()
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = X, labels = Y0)) + \
#        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002

loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002 + \
        tf.reduce_mean((Xout-X0) * (Xout - X0))

optimizer = tf.train.AdamOptimizer().minimize(loss)
optimizer_mini = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

#acc = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(X), axis=-1),\
#                                  labels=tf.argmax(Y0, axis=-1))

saver = tf.train.Saver()
loss_tr, loss_val, acc_tr, acc_val = [], [], [], []
EPOCHS, batch_size = 24, 10
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, './tfmodel1')
    batches = [img_paths[batch_size*j:batch_size*j+batch_size] for j \
         in range(int(len(img_paths)/4))]
    
    for i in range(EPOCHS):
        for i0,batch in enumerate(batches[:320]):
            
            Xz = [np.expand_dims(cv2.resize(cv2.imread(j,cv2.IMREAD_UNCHANGED)[:,:,-1],\
                (100,100), interpolation=cv2.INTER_CUBIC).ravel(), 0)*1./255 for j in batch]
            Yz = [np.expand_dims(onehot(len(os.listdir('Cyrillic')), \
                names_dict[k.split('/')[1]]),0) for k in batch]
            
            Xz = np.concatenate(Xz, axis=0)
            Yz = np.concatenate(Yz, axis=0)
            
            if i < 3:
                _, a = sess.run([optimizer, loss], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:]})#, \
                                         #Y0:Yz[:int(0.8*batch_size),:]})
            else:
                _, a = sess.run([optimizer_mini, loss], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:]})#, \
                                         #Y0:Yz[:int(0.8*batch_size),:]})
            loss_tr += [float(np.mean(a))]
            #acc_tr += [float(b)]
            
            a = sess.run([loss], \
                            feed_dict = {X0:Xz[int(0.8*batch_size):,:]})#, \
                                         #Y0:Yz[int(0.8*batch_size):,:]})    
            loss_val += [float(np.mean(a))]
            #acc_val += [float(b)]
            
            print('===================================================')
            print('Batch num', i0, 'epo num', i)
            print('Loss', loss_tr[-1], loss_val[-1])
            #print('Acc', acc_tr[-1], acc_val[-1])
            
        if (i+1) % 3 == 0:
            json.dump({'loss_val':loss_val, 'loss_tr':loss_tr, \
                       'acc_val':acc_val, 'acc_tr':acc_tr}, \
                    open('losses.json', 'w'))
            true = np.argmax(Yz, axis = -1)
            pred = sess.run([Xout], feed_dict={X0:Xz})
            #pred = list(np.argmax(pred, -1))
            print('True', 'Pred')
            print(list(zip(true, pred)))
            saver.save(sess, './tfmodel1')





