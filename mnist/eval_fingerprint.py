from __future__ import print_function
from collections import defaultdict
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import codecs
import errno
import numpy as np
import os
import os
import os.path
import pickle
import random
import scipy as sp
import sys
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import custom_datasets

sys.path.append("..")
import util
import fp_train
import fp_eval
from fingerprint import Fingerprints

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
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
parser.add_argument('--ckpt', type=str, default="/tmp/user/mnist/")
parser.add_argument('--log-dir', type=str)
parser.add_argument('--adv-ex-dir')
parser.add_argument('--fingerprint-dir')
parser.add_argument('--data-dir', type=str)

parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--num-dx', type=int, default=5)
parser.add_argument('--num-class', type=int, default=10)
#parser.add_argument('--tau', type=str, default="0.1,0.2")
parser.add_argument('--name', default="dataset-name")

util.add_boolean_argument(parser, "verbose", default=False)
util.add_boolean_argument(parser, "debug", default=False)

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
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""
test_set_path = os.path.join(args.adv_ex_dir,'Random_Test_%s_.p' % ('mnist'))
test_loader = torch.utils.data.DataLoader(
            custom_datasets.Adv(filename=test_set_path, transp=True),
            batch_size=args.batch_size, shuffle=False, **kwargs)

random_loader = torch.utils.data.DataLoader(
    custom_datasets.RandomMNIST(transform=transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)

list_advs = ["adapt-pgd"] #, "bim-a", "bim-b", "jsma", "cw-l2"]
# List of attacks, copy from run_search

dataset = 'mnist'
list_adv_loader=[]
for advs in list_advs:
    attack_file = os.path.join(args.adv_ex_dir, 'Adv_%s_%s.p' % (dataset, advs))
    adv_loader= torch.utils.data.DataLoader(
            custom_datasets.Adv(filename=attack_file, transp=True),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    list_adv_loader.append(adv_loader)

from model import CW_Net as Net
#from small_model import Very_Small_Net as Net

print("Eval using model", Net)

model = Net()
print("Loading ckpt", args.ckpt)
model.load_state_dict(torch.load(args.ckpt))

if args.cuda:
    model.cuda()
model.eval()

print("Args:", args)

fixed_dxs = pickle.load(open(os.path.join(args.fingerprint_dir, "fp_inputs_dx.pkl"), "rb"))
fixed_dys = pickle.load(open(os.path.join(args.fingerprint_dir, "fp_outputs.pkl"), "rb"))

fp = Fingerprints()
fp.dxs = fixed_dxs
fp.dys = fixed_dys

loaders = [test_loader]
loaders.extend(list_adv_loader)

names = ["test"]
names.extend(list_advs)

assert (len(names) == len(loaders))
reject_thresholds = [0. + 0.001 * i for i in range(2000)]

results = {}


data_loader = test_loader
ds_name = "test"
print("Dataset", ds_name)
test_results_by_tau, test_stats_by_tau = fp_eval.eval_with_fingerprints(model, data_loader, ds_name, fp, reject_thresholds, None, args)
results["test"] = test_results_by_tau

for data_loader, ds_name in zip(loaders, names):

    if ds_name == "test": continue
    print("Dataset", ds_name)

    results_by_tau, stats_by_tau = fp_eval.eval_with_fingerprints(model, data_loader, ds_name, fp, reject_thresholds, test_stats_by_tau, args)
    results[ds_name] = results_by_tau

# Get precision / recall where positive examples = adversarials, negative examples = real inputs.
for item,advs in enumerate(list_advs):
    print("AUC-ROC for %s",advs)
    pos_names = [advs] # advs
    neg_names = [names[0]] # test
    fp_eval.get_pr_wrapper(results, pos_names, neg_names, reject_thresholds, args)
