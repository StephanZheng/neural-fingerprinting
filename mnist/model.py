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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class CW_Net(nn.Module):
    def __init__(self):
        super(CW_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bnm1 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bnm2 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bnm3 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bnm4 = nn.BatchNorm2d(64, momentum=0.1)
        self.fc1 = nn.Linear(1024, 200)
        self.bnm5 = nn.BatchNorm1d(200, momentum=0.1)
        self.fc2 = nn.Linear(200, 200)
        self.bnm6 = nn.BatchNorm1d(200, momentum=0.1)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bnm1(out)
        out = F.relu(self.conv2(out))
        out = self.bnm2(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = self.bnm3(out)
        out = F.relu(self.conv4(out))
        out = self.bnm4(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.bnm5(out)
        out = F.relu(self.fc2(out))
        out = self.bnm6(out)
        out = self.fc3(out)
        return (out)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bnm1 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bnm2 = nn.BatchNorm2d(64, momentum=0.1)
        self.fc1   = nn.Linear(1024, 200)
        self.bnm3 = nn.BatchNorm2d(200, momentum=0.1)
        self.fc2   = nn.Linear(200, 84)
        self.bnm4 = nn.BatchNorm2d(84, momentum=0.1)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bnm1(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = self.bnm2(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.bnm3(out)
        out = F.relu(self.fc2(out))
        out = self.bnm4(out)
        out = self.fc3(out)
        return out
