from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
# import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import scipy as sp
# import sklearn as skl
import pickle

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import random
import sys
sys.path.append('..')
import util

class RandomMNIST(data.Dataset):

    def __init__(self, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img = torch.randn(28, 28)
        target = random.randint(0,9)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return 10000

class Adv(data.Dataset):

    def __init__(self, transform=None, target_transform=None, filename="adv_set_e_2.p", transp = False):
        """

        :param transform:
        :param target_transform:
        :param filename:
        :param transp: Set shuff= False for PGD based attacks
        :return:
        """
        self.transform = transform
        self.target_transform = target_transform
        self.adv_dict=pickle.load(open(filename,"rb"))
        self.adv_flat=self.adv_dict["adv_input"]
        self.num_adv=np.shape(self.adv_flat)[0]
        self.transp = transp
        self.sample_num = 0

    def __getitem__(self, index):
        img=self.adv_flat[self.sample_num,:]
        if(self.transp == False):
            # shuff is true for non-pgd attacks
            img = torch.from_numpy(np.reshape(img,(28,28)))
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor)
        target = np.argmax(self.adv_dict["adv_labels"],axis=1)[self.sample_num]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        self.sample_num = self.sample_num + 1
        return img, target

    def __len__(self):
        # Feed length as an argument
        return 14
