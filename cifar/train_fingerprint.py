from __future__ import print_function
from collections import defaultdict
from scipy.stats import ortho_group #For Generating Orthogonal Fingerprint (dys)
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import dill as pickle
import numpy as np
import os
import sys
import torch
import torch.optim as optim
sys.path.append("..")
import util

import custom_datasets
from torch.utils.data.sampler import SubsetRandomSampler
import fp_train
from fingerprint import Fingerprints

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--ds-name', type=str, default = 'cifar',
                    help='Dataset -- mnist, cifar, miniimagenet')
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

train_transform = transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((0.5, 0.5, 0.5),
                                		(1.0,1.0,1.0))])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                (1.0,1.0,1.0))])
# Create Validation Set
train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                download=True, transform=train_transform)

valid_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
download=True, transform=transform)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train)) #10% of train data is val data at 0.1
np.random.shuffle(indices)

train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(valid_indices)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler = train_sampler,
    batch_size=args.batch_size,  **kwargs)
test_loader = torch.utils.data.DataLoader(
    valid_dataset, sampler = val_sampler,
    batch_size=args.test_batch_size, **kwargs)

#random_loader = torch.utils.data.DataLoader(
#    custom_datasets.RandomCIFAR10(args.data_dir, transform=transform),
#    batch_size=args.batch_size, shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Construct fingerprint patterns

# Choose xs
fp_dx = ([(np.random.rand(1,3,32,32)-0.5)*2*args.eps for i in range(args.num_dx)])
# fp_dx = [np.zeros(1,1,28,28)*args.eps for i in range(args.num_dx)]

# for i in range(args.num_dx):
#     k,l = random.randint(0,27), random.randint(0,27)

fp_inputs_pkl = open(os.path.join(args.log_dir, "fp_inputs_dx.pkl"), "wb")
pickle.dump(fp_dx, fp_inputs_pkl)
fp_inputs_pkl.close()

# Target ys
# num_target_classes x num_perturb x num_class
fp_target = 0.254*np.ones((args.num_class, args.num_dx, args.num_class))

for j in range(args.num_dx):
    for i in range(args.num_class):
        fp_target[i,j,i] = - 0.7

fp_target = 1.5*fp_target
fp_target_pkl = open(os.path.join(args.log_dir, "fp_outputs.pkl"), "wb")
pickle.dump(fp_target, fp_target_pkl)
fp_target_pkl.close()

fp_target = util.np2var(fp_target, args.cuda)

fp = Fingerprints()
fp.dxs = fp_dx
fp.dys = fp_target

from model import CW2_Net as Net
#from res_model import ResNet as Net
from models import *

print("Train using model", Net)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=args.momentum)

print("Args:", args)

val_losses=[]
for epoch in range(1, args.epochs + 1):
    if(epoch==1):
        test_loss = fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys, test_length=0.1*len(valid_dataset))
    fp_train.train(epoch, args, model, optimizer, train_loader, fp.dxs, fp.dys)
    test_loss = fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys, test_length=0.1*len(valid_dataset))
    val_losses.append(test_loss)
    loss_flag = 1
    #for i in range(2,5):
    #    if(epoch<=15 or val_losses[-1]<val_losses[-i]):
    #       loss_flag = 1

    path = os.path.join(args.log_dir, "ckpt", "state_dict-ep_{}.pth".format(epoch))
    print("Saving model in", path)
    torch.save(model.state_dict(), path)

    if(not loss_flag == 1):
        with open(os.path.join(args.log_dir, "termination_epoch"), "wb") as f:
            f.write(str(epoch)+"\n")
            print("Saving termination epoch number in {}".format(os.path.join(args.log_dir, "termination_epoch")))
        break
    elif(epoch == args.epochs):
        term_epoch = np.argmin(val_losses) + 1
        with open(os.path.join(args.log_dir, "termination_epoch"), "wb") as f:
            f.write(str(term_epoch)+"\n")
            print("Saving termination epoch number in {}".format(os.path.join(args.log_dir, "termination_epoch")))

