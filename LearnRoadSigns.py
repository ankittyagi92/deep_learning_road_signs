# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 00:53:58 2017

@author: ankit_lr03y9t
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import  SGD
from keras.utils import np_utils
from keras import backend as K

#set directories
os.getcwd()
os.chdir("C:/Users/ankit_lr03y9t/Documents/Deep")

#load train and view
train=pickle.load(open("training_set.pkl", "rb"))
plt.imshow(train[28]["img"])
train[28]["img"].shape
len(train)

#extract image and class in separate lists
train_image=[]
train_class=[]
for ele in train:
    train_image.append(ele["img"])
    train_class.append(ele["class"])

plt.imshow(train_image[28],cmap='nipy_spectral')
train_image[28].shape

#plot distributon of data across classes
train_class_np = np.array(train_class)
plt.bar(range(len(np.bincount(train_class_np))),np.bincount(train_class_np))
# some classes show very few samples
n_classes=max(train_class)+1

#reshaping input images
a=train[28]["img"]
plt.imshow(a)
b=imresize(a,(40,40))
plt.imshow(b)

#find average dimension of images in training set
sum_h=0
sum_w=0
for i in range(0,len(train_image)):
    sum_h+=train_image[i].shape[0]
    sum_w+=train_image[i].shape[1]
print("average input dimensions are: {} X {}".format(sum_h/len(train_image)\
                                                    , sum_w/len(train_image)))    
# average size was around 50X50
#im_w,im_h=50,50
                                                    
                                                    
# image size of 50X50 had huge runtime, reducing to 25X25
im_w,im_h=25,25

#resize all image in train
for i in range(0,len(train_image)):
    train_image[i] =imresize(train_image[i],(im_w,im_h))
#noramlize all images in train
for i in range(0,len(train_image)):
    train_image[i] =train_image[i]/255.

#not using stratified splitting for now.
#ideally all the classes should have similar no. of observations against them
#for this classes with less observation should be populated using transformations
#on their existing data, like translation, rotation etc

#ideally a stratified split, or at least a randomised split should be used
#splitting directly for now
#split the train 70-30 for modeling
train_img= np.array(train_image[0:27446])
test_img= np.array(train_image[27446:39209])
train_cl= np.array(train_class_np[0:27446])
test_cl= np.array(train_class_np[27446:39209])

#set shape according to dim ordering
if K.image_dim_ordering() == 'th':
    train_img = train_img.reshape(train_img.shape[0], 3, im_h, im_w)
    test_img = test_img.reshape(test_img.shape[0], 3, im_h, im_w)
    input_shape = (3, im_h, im_w)
else:
    train_img = train_img.reshape(train_img.shape[0], im_h, im_w, 3)
    test_img = test_img.reshape(test_img.shape[0], im_h, im_w, 3)
    input_shape = (im_h, im_w, 3)
    
print("input shape is: {}".format(input_shape))
print("train shape is: {}".format(train_img.shape))
print("test shape is: {}".format(test_img.shape))

#one-hot vector for classes
train_cl = np_utils.to_categorical(train_cl, nb_classes=n_classes)
test_cl = np_utils.to_categorical(test_cl, nb_classes=n_classes)

#Model parameters
#using mnist model for first run

#hyperparameteres
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#compile model
sgd = SGD(lr=1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
              
#run model
batch_size = 128
#using 6 epochs only, for runtime considerations
nb_epoch = 6


model.fit(train_img, train_cl, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test_img, test_cl))