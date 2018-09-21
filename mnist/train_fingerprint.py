from __future__ import print_function
from collections import defaultdict
#from scipy.stats import ortho_group #For Generating Orthogonal Fingerprint (dys)
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import dill as pickle
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import util

import custom_datasets

import fp_train
from fingerprint import Fingerprints

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--ds-name', type=str, default = 'mnist',
                    help='Dataset -- mnist, cifar, miniimagenet')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--num-dx', type=int, default=5)
parser.add_argument('--num-class', type=int, default=10)

parser.add_argument('--name', default="dataset-name")
parser.add_argument("--data-dir")
parser.add_argument("--log-dir")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=True, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



# Construct fingerprint patterns

# Choose xs
fp_dx = [np.random.rand(1,1,28,28)*args.eps for i in range(args.num_dx)]
# fp_dx = [np.zeros(1,1,28,28)*args.eps for i in range(args.num_dx)]

# for i in range(args.num_dx):
#     k,l = random.randint(0,27), random.randint(0,27)

pickle.dump(fp_dx, open(os.path.join(args.log_dir, "fp_inputs_dx.pkl"), "wb"))

# Target ys
# num_target_classes x num_perturb x num_class
fp_target = -0.2357*np.ones((args.num_class, args.num_dx, args.num_class))

for j in range(args.num_dx):
    for i in range(args.num_class):
        fp_target[i,j,i] = 0.7

pickle.dump(fp_target, open(os.path.join(args.log_dir, "fp_outputs.pkl"), "wb"))

fp_target = util.np2var(fp_target, args.cuda)

fp = Fingerprints()
fp.dxs = fp_dx
fp.dys = fp_target


#from model import Net
from model import CW_Net as Net
#from small_model import Very_Small_Net as Net

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Args:", args)

for epoch in range(1, args.epochs + 1):
    if(epoch==1):
        fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys)
    fp_train.train(epoch, args, model, optimizer, train_loader, fp.dxs, fp.dys)
    fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys)

    path = os.path.join(args.log_dir, "ckpt", "state_dict-ep_{}.pth".format(epoch))
    print("Saving model in", path)
    torch.save(model.state_dict(), path)
