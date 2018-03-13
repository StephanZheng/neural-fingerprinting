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
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
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
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1600, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return (out)

class CW2_Net(nn.Module):
    def __init__(self):
        super(CW2_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bnm1 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnm2 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnm3 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bnm4 = nn.BatchNorm2d(128, momentum=0.1)
        self.fc1 = nn.Linear(3200, 256)
        #self.dropout1 = nn.Dropout(p=0.35, inplace=False)
        self.bnm5 = nn.BatchNorm1d(256, momentum=0.1)
        self.fc2 = nn.Linear(256, 256)
        self.bnm6 = nn.BatchNorm1d(256, momentum=0.1)
        self.fc3 = nn.Linear(256, 10)
        #self.dropout2 = nn.Dropout(p=0.35, inplace=False)
        #self.dropout3 = nn.Dropout(p=0.35, inplace=False)

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
        #out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        #out = self.dropout2(out)
        out = self.bnm5(out)
        out = F.relu(self.fc2(out))
        #out = self.dropout3(out)
        out = self.bnm6(out)
        out = self.fc3(out)
        return (out)

class LID_Net(nn.Module):
    def __init__(self):
        super(LID_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(128, 1024)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = F.relu(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        return (out)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return (out)
