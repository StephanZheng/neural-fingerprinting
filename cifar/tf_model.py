from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
K.set_image_data_format('channels_first')
from models import *
from model import *

class Model(object):
  def __init__(self, torch_model=CW2_Net,softmax=True):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 3, 32, 32])

    # Model from https://arxiv.org/pdf/1608.04644.pdf
    if(isinstance(torch_model,CW2_Net)):
      model = Sequential()
      model.add(Conv2D(32, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=(3, 32, 32),
                       name='conv1'))
      model.add(BatchNormalization(axis=1,name='bnm1',momentum=0.1))
      model.add(Conv2D(64, (3, 3),activation='relu',name='conv2'))
      model.add(BatchNormalization(axis=1,name='bnm2',momentum=0.1))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(128, (3, 3), activation='relu', name='conv3'))
      model.add(BatchNormalization(axis=1,name='bnm3',momentum=0.1))
      model.add(Conv2D(128, (3, 3), activation='relu', name='conv4'))
      model.add(BatchNormalization(axis=1,name='bnm4',momentum=0.1))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Flatten())
      model.add(Dense(256, activation='relu', name='fc1'))
      model.add(BatchNormalization(axis=-1,name='bnm5',momentum=0.1))
      model.add(Dense(256, activation='relu', name='fc2'))
      model.add(BatchNormalization(axis=1,name='bnm6',momentum=0.1))
      model.add(Dense(10, activation=None, name='fc3'))

    elif(isinstance(torch_model,LeNet)):
      model = Sequential()
      model.add(Conv2D(6, kernel_size=(5, 5),
                       activation='relu',
                       input_shape=(3, 32, 32),
                       name='conv1'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(16, (5, 5), activation='relu', name='conv2'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Flatten())
      model.add(Dense(120, activation='relu', name='fc1'))
      model.add(Dense(84, activation='relu', name='fc2'))
      model.add(Dense(10, activation=None, name='fc3'))

    elif(isinstance(torch_model,LID_Net)):
      model = Sequential()
      model.add(Conv2D(32, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=(3, 32, 32),
                       name='conv1'))
      model.add(Conv2D(32, (3, 3),activation='relu',name='conv2'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(64, (3, 3), activation='relu', name='conv3'))
      model.add(Conv2D(64, (3, 3), activation='relu', name='conv4'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(128, (3, 3), activation='relu', name='conv5'))
      model.add(Conv2D(128, (3, 3), activation='relu', name='conv6'))
      model.add(Flatten())
      #model.add(Dropout(rate=0.5))
      model.add(Dense(1024, activation='relu', name='fc1'))
      #model.add(Dropout(rate=0.5))
      model.add(Dense(512, activation='relu', name='fc2'))
      #model.add(Dropout(rate=0.5))
      model.add(Dense(10, activation=None, name='fc3'))

    if(softmax==True):
      # Set softmax=False for CW-l2 attack
      model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    self.logits = model(self.x_image)
    y_ = self.y_input
    y_conv=self.logits
    x=self.x_input
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=self.y_input, logits=self.logits)

    self.xent = tf.reduce_sum(y_xent)
    self.y_pred = tf.argmax(self.logits, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.model = model

  @staticmethod


  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

  @staticmethod
  def _weight_variable(shape,name=None):
    """weight_variable generates a weigh  t        variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.01)+0.01
    return tf.Variable(initial,name=name)

  @staticmethod
  def _bias_variable1(shape,name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial,name=name)

  @staticmethod
  def _bias_variable(shape,name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial,name=name)

