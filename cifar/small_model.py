from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=2)
        #self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        #self.drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(784, 200)
        self.fc1 = nn.Linear(360, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return (x, F.log_softmax(x))


class Very_Small_Net(nn.Module):
    def __init__(self):
        super(Very_Small_Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2)
        #self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=2)
        #self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        #self.drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(784, 200)
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return (x, F.log_softmax(x))
"""
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
K.set_image_data_format('channels_first')
def lenet_keras():

    model = Sequential()
    model.add(Conv2D(10, kernel_size=(3, 3),
                     strides=(2,2), activation='relu',
                     input_shape=(1,28,28),
                     name='conv1'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, kernel_size=(3, 3), strides=(2,2),
                     activation='relu', name='conv2'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dense(120, activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc1'))
    model.add(Dense(10, activation=None, name='fc2'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta())

    return model
"""
