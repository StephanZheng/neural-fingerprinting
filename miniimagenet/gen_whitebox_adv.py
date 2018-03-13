"""
Parts of the code have been copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch
import pickle
sys.path.append("..")
import util
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
from torch.utils.data.sampler import SubsetRandomSampler

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
and callable(models.__dict__[name]))
# Training settings
parser = argparse.ArgumentParser(description='MiniImageNet BenchMarkSetUp')
parser.add_argument('--data', metavar='DIR',
                    default='miniimagenet/data/miniImagenet/test_imnet'
                    ,help='path to dataset')
parser.add_argument('--test_indices_file', metavar='DIR',
                    default='/home/user/Documents/My_Software/miniim/adversarial/test_indices.p'
                    ,help='path to indices of (unseen) samples')
parser.add_argument('--ckpt', default="/home/user/Documents/My_Software/miniim/" \
                        + "user/logs/miniimagenet/eps_0.03/numdx_20/ckpt/state_dict-ep_29.pth")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

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

train_transform = transform = transforms.Compose([#transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
						                transforms.Normalize((0.5, 0.5, 0.5),
                                		(1.0,1.0,1.0))])

transform = transforms.Compose([ transforms.RandomResizedCrop(32),
                                 transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                (1.0,1.0,1.0))])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

from modelz import alexnet
#from small_model import Very_Small_Net as Net
ckpt = args.ckpt
test_image_dir = args.data
test_indices_file = args.test_indices_file
model = alexnet.alexnet()
print("Loading ckpt", ckpt)
model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))
model.train(False)
model.eval()
if args.cuda:
    model.cuda()

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


transform = transforms.Compose([ transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                (1.0,1.0,1.0))])


test_dataset = datasets.ImageFolder(
        test_image_dir,
        transform=transform,)

test_loader = pickle.load(open(test_indices_file,"rb"))
from third_party.attack_iterative import *
import numpy as np
dataset = "miniimagenet20"
#attack = "iterativefgsm"
attack = "fgsm"

adv_gen = AttackIterative(targeted = False,
        max_epsilon=16.0,
        norm=float('inf'),
        step_alpha=0.0,
        num_steps=1,
        cuda=False,
        debug=False)
yy = np.zeros((1,20))
yy[0,5] = 1
pytorch_model = model
advs = None
total_correct = 0
for batch_idx, (x, y) in enumerate(test_loader):
    #x = Variable(x)
    print("batch", batch_idx)
    output = pytorch_model(Variable(x))
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(Variable(y).data.view_as(pred)).cpu().sum()
    total_correct = total_correct + correct

    print(total_correct)

    if(advs is None):
        x_test = x.numlspy()
        y_test = y.numpy()
        advs = adv_gen.run(pytorch_model, x, y)
        output = pytorch_model(Variable(torch.from_numpy(advs)))
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(Variable(y).data.view_as(pred)).cpu().sum()
        advs = torch.from_numpy(advs)
        total_correct = total_correct + correct
    else:
        new_advs = adv_gen.run(pytorch_model, x, y)
        advs = np.concatenate((advs, new_advs))
        y_test = np.concatenate((y_test,y.numpy()))
        x_test = np.concatenate((x_test,x.numpy()))

    f = open('Adv_%s_%s_%s.p' % (dataset, attack,batch_idx),'w')
    pickle.dump({"adv_input":advs,"adv_labels":y_test},f)
    f.close()

    f = open('Random_Test_%s_%s.p' % (dataset,batch_idx),'w')
    pickle.dump({"adv_input":x_test,"adv_labels":y_test},f)
    f.close()



