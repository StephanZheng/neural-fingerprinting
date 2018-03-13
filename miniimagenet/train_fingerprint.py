from __future__ import print_function
"""
Parts of the code have been copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
from collections import defaultdict
from scipy.stats import ortho_group #For Generating Orthogonal Fingerprint (dys)
from torch.autograd import Variable
from torchvision import datasets, transforms, models
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
import fp_train
from fingerprint import Fingerprints
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import pickle 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
and callable(models.__dict__[name]))
# Training settings
parser = argparse.ArgumentParser(description='MiniImageNet BenchMarkSetUp')
parser.add_argument('--data', metavar='DIR',
                    default='miniimagenet/data/miniImagenet/test_imnet'
                    ,help='path to dataset')
parser.add_argument('--ds-name', type=str, default = 'mnist',
                    help='Dataset -- mnist, cifar, miniimagenet')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                    ' (default: resnet18)')
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

train_transform = transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
						                transforms.Normalize((0.5, 0.5, 0.5),
                                		(1.0,1.0,1.0))])

transform = transforms.Compose([ transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                (1.0,1.0,1.0))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
traindir = args.data
valdir = os.path.join(args.data, 'val')
testdir = os.path.join(args.data,'test')

train_dataset = datasets.ImageFolder(
        traindir,
         transform = transform)

valid_dataset = datasets.ImageFolder(
        traindir,
        transform=transform,)

test_dataset = datasets.ImageFolder(
        traindir,
        transform=transform,)

num_train = len(train_dataset)
indices = list(range(num_train))

for i in range(100):
    np.random.shuffle(indices)

split1 = int(np.floor(0.14 * num_train)) #10% of train data is val data at 0.1
split2 = int(np.floor(0.23 * num_train)) #10% of train data is val data at 0.1


train_indices, valid_indices, test_indices = indices[split2:], indices[split1:split2], indices[:split1]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SequentialSampler(valid_indices)
test_sampler = SequentialSampler(test_indices)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler = train_sampler,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    train_dataset, sampler = val_sampler,
    batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

test_loader2 = torch.utils.data.DataLoader(
    train_dataset, sampler = test_sampler,
    batch_size=args.test_batch_size, shuffle=False, pin_memory=False)

with open("test_indices.p","wb") as f1:
    pickle.dump(test_loader2, f1)


# Construct fingerprint patterns

# Choose xs
fp_dx = ([(np.random.rand(1,3,224,224)-0.5)*2*args.eps for i in range(args.num_dx)])
# fp_dx = [np.zeros(1,1,28,28)*args.eps for i in range(args.num_dx)]

# for i in range(args.num_dx):
#     k,l = random.randint(0,27), random.randint(0,27)

fp_inputs_pkl = open(os.path.join(args.log_dir, "fp_inputs_dx.pkl"), "wb")
pickle.dump(fp_dx, fp_inputs_pkl)
fp_inputs_pkl.close()

# Target ys
# num_target_classes x num_perturb x num_class
fp_target = -0.254*np.ones((args.num_class, args.num_dx, args.num_class))

for j in range(args.num_dx):
    for i in range(args.num_class):
        fp_target[i,j,i] = 0.6
        #fp_target[i,j,(i+1)%args.num_class] = 0.6
fp_target =  1.0*fp_target
fp_target_pkl = open(os.path.join(args.log_dir, "fp_outputs.pkl"), "wb")
pickle.dump(fp_target, fp_target_pkl)
fp_target_pkl.close()

fp_target = util.np2var(fp_target, args.cuda)

fp = Fingerprints()
fp.dxs = fp_dx
fp.dys = fp_target


#model = models.__dict__['vgg19']()
#from model import CW_Net as Net
from modelz import alexnet
#model = Net()
model = alexnet.alexnet()
args.distributed = False
if not args.distributed and not args.no_cuda:
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
elif not args.no_cuda:
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)


optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=args.momentum)

print("Args:", args)

for epoch in range(1, args.epochs + 1):
    if(epoch==1):
        pass
        #fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys)
    fp_train.train(epoch, args, model, optimizer, train_loader, fp.dxs, fp.dys)
    fp_train.test(epoch, args, model, test_loader, fp.dxs, fp.dys)
    fp_train.test(epoch, args, model, test_loader2, fp.dxs, fp.dys)
    path = os.path.join(args.log_dir, "ckpt", "state_dict-ep_{}.pth".format(epoch))
    print("Saving model in", path)
    torch.save(model.state_dict(), path)
    
