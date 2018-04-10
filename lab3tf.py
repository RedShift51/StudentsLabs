#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt, os, json, cv2
import tensorflow as tf
import time
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
X0 = tf.placeholder(dtype=tf.float32, shape=[None,100,100,1])
Y0 = tf.placeholder(dtype=tf.float32, shape = [None,len(os.listdir('Cyrillic'))])

""" Net's architecture """
def net0(X0):
    
    stage1 = tf.contrib.layers.conv2d(X0, num_outputs=48, kernel_size=(7,7), \
            padding='SAME', activation_fn=None, stride=(2,2))
    stage2 = tf.contrib.layers.max_pool2d(stage1, kernel_size=(4,4))
    stage3 = tf.contrib.layers.conv2d(stage2, num_outputs=48, kernel_size=(4,4), \
            padding='SAME', activation_fn=None)
    stage4 = tf.contrib.layers.conv2d(stage3, num_outputs=48, kernel_size=(4,4), \
            padding='SAME', activation_fn=None)
    stage5 = tf.contrib.layers.max_pool2d(stage4, kernel_size=(3,3))
    
    stage6 = tf.contrib.layers.conv2d(stage5, num_outputs=48, kernel_size=(4,4), \
            padding='SAME', activation_fn=None)
    stage7 = tf.contrib.layers.conv2d(stage6, num_outputs=48, kernel_size=(3,3), \
            padding='SAME', activation_fn=None)
    stage8 = tf.contrib.layers.max_pool2d(stage7, kernel_size=(3,3))
    
    fl1 = tf.contrib.layers.flatten(stage8)
    dense1 = tf.contrib.layers.fully_connected(fl1, num_outputs=300, activation_fn=None)
    dense2 = tf.contrib.layers.fully_connected(dense1, num_outputs=300, activation_fn=None)
    
    out = tf.contrib.layers.fully_connected(dense2, \
                num_outputs=len(os.listdir('Cyrillic')), \
                activation_fn=None)    

    return out

def net1(X0):
    
    stage1 = tf.contrib.layers.conv2d(X0, num_outputs=48, kernel_size=(7,7), \
            padding='SAME', stride=(2,2))
    stage2 = tf.contrib.layers.max_pool2d(stage1, kernel_size=(4,4))
    stage3 = tf.contrib.layers.conv2d(stage2, num_outputs=48, kernel_size=(4,4), \
            padding='SAME')
    stage4 = tf.contrib.layers.conv2d(stage3, num_outputs=48, kernel_size=(4,4), \
            padding='SAME')
    stage5 = tf.contrib.layers.max_pool2d(stage4, kernel_size=(3,3))
    
    fl1 = tf.contrib.layers.flatten(stage5)
    dense1 = tf.contrib.layers.fully_connected(fl1, num_outputs=300)
    dense2 = tf.contrib.layers.fully_connected(dense1, num_outputs=300)
    
    out = tf.contrib.layers.fully_connected(dense2, \
                num_outputs=len(os.listdir('Cyrillic')), \
                activation_fn=None)    

    return out

def net2(X0):
    
    stage1 = tf.contrib.layers.conv2d(X0, num_outputs=48, kernel_size=(7,7), \
            padding='SAME', stride=(2,2))
    stage2 = tf.contrib.layers.max_pool2d(stage1, kernel_size=(4,4))
    
    fl1 = tf.contrib.layers.flatten(stage2)
    dense1 = tf.contrib.layers.fully_connected(fl1, num_outputs=300)
    dense2 = tf.contrib.layers.fully_connected(dense1, num_outputs=300)
    
    out = tf.contrib.layers.fully_connected(dense2, \
                num_outputs=len(os.listdir('Cyrillic')), \
                activation_fn=None)    

    return out



X1 = net2(X0)
#X2 = net1(X0)
#X3 = net2(X0)
#X = tf.clip_by_value(X, -10, 10)

vars = tf.trainable_variables()
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = X1, labels = Y0)) + \
        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002
#loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = X2, labels = Y0)) + \
#        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002
#loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = X3, labels = Y0)) + \
#        tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002

#loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.002 + \
#        tf.reduce_mean((Xout-X0) * (Xout - X0))

optimizer1 = tf.train.AdamOptimizer().minimize(loss1)
optimizer_mini1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss1)

#optimizer2 = tf.train.AdamOptimizer().minimize(loss2)
#optimizer_mini2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss2)

#optimizer3 = tf.train.AdamOptimizer().minimize(loss3)
#optimizer_mini3 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss3)

acc1 = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(X1), axis=-1),\
                                  labels=tf.argmax(Y0, axis=-1))
#acc2 = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(X2), axis=-1),\
#                                  labels=tf.argmax(Y0, axis=-1))
#acc3 = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(X3), axis=-1),\
#                                  labels=tf.argmax(Y0, axis=-1))

saver = tf.train.Saver()
loss_tr1, loss_val1, acc_tr1, acc_val1 = [], [], [], []
loss_tr2, loss_val2, acc_tr2, acc_val2 = [], [], [], []
loss_tr3, loss_val3, acc_tr3, acc_val3 = [], [], [], []
t1, t2, t3 = [], [], []
EPOCHS, batch_size = 12, 40
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, './tfmodel1')
    batches = [img_paths[batch_size*j:batch_size*j+batch_size] for j \
         in range(int(len(img_paths)/4))]
    start = time.time()
    for i in range(EPOCHS):
        for i0,batch in enumerate(batches[:320]):
            
            Xz = [np.expand_dims(cv2.resize(cv2.imread(j,cv2.IMREAD_UNCHANGED)[:,:,-1],\
                (100,100), interpolation=cv2.INTER_CUBIC), 0)*1./255 for j in batch]
            Yz = [np.expand_dims(onehot(len(os.listdir('Cyrillic')), \
                names_dict[k.split('/')[1]]),0) for k in batch]
            
            Xz = np.expand_dims(np.concatenate(Xz, axis=0),-1)
            Yz = np.concatenate(Yz, axis=0)
            

            if i < 16:
                _, a, b = sess.run([optimizer1, loss1, acc1], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            else:
                _, a, b = sess.run([optimizer_mini1, loss1, acc1], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            loss_tr1 += [float(np.mean(a))]
            acc_tr1 += [float(b)]
            
            a, b = sess.run([loss1, acc1], \
                            feed_dict = {X0:Xz[int(0.8*batch_size):,:], \
                                         Y0:Yz[int(0.8*batch_size):,:]})    
            loss_val1 += [float(np.mean(a))]
            acc_val1 += [float(b)]
            
            ############################################################
            """
            if i < 16:
                _, a, b = sess.run([optimizer2, loss2, acc2], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            else:
                _, a, b = sess.run([optimizer_mini2, loss2, acc2], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            loss_tr2 += [float(np.mean(a))]
            acc_tr2 += [float(b)]
            
            a, b = sess.run([loss2, acc2], \
                            feed_dict = {X0:Xz[int(0.8*batch_size):,:], \
                                         Y0:Yz[int(0.8*batch_size):,:]})    
            loss_val2 += [float(np.mean(a))]
            acc_val2 += [float(b)]
            """
            ##############################################################
            """
            if i < 16:
                _, a, b = sess.run([optimizer3, loss3, acc3], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            else:
                _, a, b = sess.run([optimizer_mini3, loss3, acc3], \
                            feed_dict = {X0:Xz[:int(0.8*batch_size),:], \
                                         Y0:Yz[:int(0.8*batch_size),:]})
            loss_tr3 += [float(np.mean(a))]
            acc_tr3 += [float(b)]
            
            a, b = sess.run([loss3, acc3], \
                            feed_dict = {X0:Xz[int(0.8*batch_size):,:], \
                                         Y0:Yz[int(0.8*batch_size):,:]})    
            loss_val3 += [float(np.mean(a))]
            acc_val3 += [float(b)]
            """
            
            print('===================================================')
            print('Batch num', i0, 'epo num', i)
            print('Loss', loss_tr1[-1], loss_val1[-1])
            print('Acc', acc_tr1[-1], acc_val1[-1])
        """    
        if (i+1) % 3 == 0:
            #json.dump({'loss_val':loss_val, 'loss_tr':loss_tr, \
            #           'acc_val':acc_val, 'acc_tr':acc_tr}, \
            #        open('losses.json', 'w'))
            true = np.argmax(Yz, axis = -1)
            pred = sess.run([X], feed_dict={X0:Xz})
            pred = list(np.argmax(pred, -1))
            print('True', 'Pred')
            print(list(zip(true, pred)))
            saver.save(sess, './tfmodel1')
        """
end = time.time()
print((end-start)/60.)






