#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np, os, cv2, matplotlib.pyplot as plt
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.optimizers import Adam

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
bn_axis=-1
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(weights='imagenet'):
    input_shape = (100,100,3)
    img_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    #x = AveragePooling2D((2, 2), name='avg_pool')(x)
    
    x = Flatten()(x)
    x = Dense(110, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    
    model = Model(img_input, x, name='resnet50')
    
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
    
    #weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #                                WEIGHTS_PATH,
    #                                cache_subdir='models',
    #                                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    
    model.load_weights(weights_path, by_name=True)
    return model

def onehot(length, num):
    ans = np.zeros((length,))
    ans[num] = 1.
    return ans

model = ResNet50()
model.summary()
"""
def tail_model():
    input_shape = (3, 3, 2048)
    img_input = Input(input_shape)
    x = Flatten()(x)
    x = Dense(110, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(img_input, x, name='tail')
    return model

tail = tail_model()
"""
names = os.listdir('/home/alex/Desktop/train')
cats = [i for i in names if i.find('cat') != -1]
dogs = [i for i in names if i.find('dog') != -1]
batches = [[cats[i], dogs[i]] for \
           i in range(int(len(dogs)))]
"""
img_paths = []
for i0,i in enumerate(os.listdir('Cyrillic')):
    img_paths += ['Cyrillic'+'/'+i+'/'+j for j in os.listdir('Cyrillic/'+i)]
        
img_paths = np.array(img_paths)
np.random.shuffle(img_paths)
np.random.shuffle(img_paths)
img_paths = list(img_paths)
names_dict = sorted(list(np.unique([i.split('/')[1] for i in img_paths])))
names_dict = {names_dict[i]:i for i in range(len(names_dict))}

batch_size = 10
batches = [img_paths[batch_size*j:batch_size*j+batch_size] for j \
               in range(int(len(img_paths)/batch_size))]
"""

opt = Adam()
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 10
history = []
for i in range(EPOCHS):
    for j0,j in enumerate(batches):
        
        pic1 = cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[0]),(100,100))/255.        
        pic2 = cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[1]),(100,100))/255.        
        #pic3 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[2], 0),(40,40))-125.)/8.        
        #pic4 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[3], 0),(40,40))-125.)/8.        
        #pic5 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[4], 0),(40,40))-125.)/8.        
        #pic6 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[5], 0),(40,40))-125.)/8.        
        #pic7 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[6], 0),(40,40))-125.)/8.        
        
        #pic8 = (cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[7], 0),(40,40))-125.)/8.
        #pic9 = cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[6]),(50,50))/255.        
        #pic10 = cv2.resize(cv2.imread('/home/alex/Desktop/train/'+j[7]),(50,50))/255.
        """
        """
        Xz = np.concatenate([np.expand_dims(pic1,0), \
                             np.expand_dims(pic2,0)],0)
        Yz = np.array([[1.,0.],[0.,1.]])
        
        #history.append(model.fit(inp1, y, validation_split=0.2))
        
        """ Here we will resize to the nearest square and then we will crop it """
        """
        Xz = [np.expand_dims(cv2.resize(cv2.imread(j,cv2.IMREAD_UNCHANGED)[:,:,-1],\
            (50,50), interpolation=cv2.INTER_CUBIC), 0)*1./255 for j in batch]
        Yz = [np.expand_dims(onehot(len(os.listdir('Cyrillic')), \
                    names_dict[k.split('/')[1]]),0) for k in batch]
        """
        #Xz = np.concatenate(Xz, axis=0)
        #Yz = np.concatenate(Yz, axis=0)
        history.append(model.fit(Xz, Yz,  epochs=1))
        print('=============================')
        print(j0, 'Epoch is', i)
    if (i+1) % 3 == 0:
        model.save('0_resnet')
        
tr_acc, val_acc = [], []
tr_loss, val_loss = [], []
for i in range(len(history)):
    tr_acc += history[i].history['acc']
    tr_loss += history[i].history['loss']
    val_acc += history[i].history['val_acc']
    val_loss += history[i].history['val_loss']

