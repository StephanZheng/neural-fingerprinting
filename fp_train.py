from __future__ import print_function
from collections import defaultdict
from scipy.stats import ortho_group #For Generating Orthogonal Fingerprint (dys)
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import dill as pickle
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util

from fingerprint import Example, Fingerprints

def train(epoch, args, model, optimizer, data_loader, fp_dx, fp_target, ds_name = None):

    fingerprint_accuracy = []

    loss_n = torch.nn.MSELoss()

    for batch_idx, (x, y) in enumerate(data_loader):
        model.train()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        optimizer.zero_grad()
        real_bs = y.size(0)
        loss_func = nn.CrossEntropyLoss()

        # Batch x args.num_dx x output_size
        fp_target_var = torch.index_select(fp_target, 0, y)

        ## Add loss for (y+dy,model(x+dx)) for each sample, each dx
        data_np = util.var2np(x, args.cuda)
        x_net = x
        for i in range(args.num_dx):
            dx = fp_dx[i]
            fp_target_var_i = fp_target_var[:,i,:]
            dx = util.np2var(dx,args.cuda)
            x_net = torch.cat((x_net,x+dx))

        logits_net = model(x_net)
        output_net = F.log_softmax(logits_net)

        yhat = output_net[0:real_bs]
        logits = logits_net[0:real_bs]
        logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)
        loss_fingerprint_y = 0
        loss_fingerprint_dy = 0
        loss_vanilla = loss_func(yhat, y)

        for i in range(args.num_dx):
            dx = fp_dx[i]
            fp_target_var_i = fp_target_var[:,i,:]
            logits_p = logits_net[(i+1)*real_bs:(i+2)*real_bs]
            logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)
            diff_logits_p = logits_p_norm - logits_norm + 0.00001
            #diff_logits_p = diff_logits_p * torch.norm(diff_logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)
            loss_fingerprint_y += loss_n(logits_p_norm, fp_target_var_i)
            loss_fingerprint_dy += loss_n(diff_logits_p, fp_target_var_i)
            
        if(ds_name == "cifar"):
            if(epoch>=0):
                loss = loss_vanilla + (1.0+50.0/args.num_dx)*loss_fingerprint_dy # + loss_fingerprint_y
            else:
                loss = loss_vanilla
        else:
            if(epoch>=0):
                loss = loss_vanilla + 1.0*loss_fingerprint_dy # + loss_fingerprint_y
            else:
                loss = loss_vanilla
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss vanilla: {:.3f} fp-y: {:.3f} fp-dy: {:.3f} Total Loss: {:.3f}'.format(
                epoch, batch_idx * len(x), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss_vanilla.data[0],
                loss_fingerprint_y.data[0],
                loss_fingerprint_dy.data[0],
                loss.data[0]))

def get_majority(votes_dict):
    real_bs=len(votes_dict.keys())
    majority=np.zeros(real_bs)
    for sample_num in range(real_bs):
        max_class=0
        for i in votes_dict[sample_num].keys():
            if(votes_dict[sample_num][i]>votes_dict[sample_num][max_class]):
                max_class=i
            pass
        majority[sample_num]=max_class
        if(votes_dict[sample_num][max_class]<=2):
            print(votes_dict[sample_num][max_class])
    return majority


def test(epoch, args, model, data_loader, fp_dx, fp_target, test_length=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct_fp = 0
    fingerprint_accuracy = []

    loss_y = 0
    loss_dy = 0
    num_same_argmax = 0

    for e,(data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)

        data_np = util.var2np(data, args.cuda)
        real_bs = data_np.shape[0]

        logits = model(data)
        output = F.log_softmax(logits)
        logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)


        fp_target_var = torch.index_select(fp_target, 0, target)

        # votes_dict = defaultdict(lambda: {i:0 for i in range(args.num_class)})
        loss_n = torch.nn.MSELoss()
        for i in range(args.num_dx):
            dx = fp_dx[i]
            fp_target_var_i = fp_target_var[:,i,:]

            logits_p = model(data + util.np2var(dx, args.cuda))
            output_p = F.log_softmax(logits_p)
            logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)

            logits_p_class = logits_p_norm.data.max(0, keepdim=True)[1]

            diff = logits_p_norm - logits_norm
            diff_class = diff.data.max(1, keepdim=True)[1]
            #diff = diff * torch.norm(diff, 2, 1, keepdim=True).reciprocal().expand(real_bs, args.num_class)

            fp_target_class = fp_target_var_i.data.max(1, keepdim=True)[1]
            loss_y += loss_n(logits_p_norm, fp_target_var_i)
            loss_dy += 10.0*loss_n(diff, fp_target_var_i)
            num_same_argmax += torch.sum(diff_class == fp_target_class)

            # for sample_num in range(real_bs):
            #     fingerprint_class = np.argmax(util.var2np(diff, args.cuda)[sample_num,:])
            #     fingerprinted_class = np.argmax(fp_target_var[i,sample_num,:])
            #     fingerprint_accuracy.append((fingerprint_class,fingerprinted_class))

            #     votes_dict[sample_num][fingerprint_class] += 1
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # print(e, "pred:", pred, "label:", target, correct)
        # pred_fp = torch.from_numpy(get_majority(votes_dict).astype(int))
        # correct_fp += pred_fp.eq(target.data.view_as(pred_fp)).cpu().sum()
    if(test_length is None):
        test_length = len(data_loader.dataset)
    test_loss /= test_length

    loss_y /= test_length
    loss_dy /= test_length
    argmax_acc = num_same_argmax*1.0 / (test_length * args.num_dx)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, test_length,
        100. * correct / test_length))

    # print('\nTest set: Average loss: {:.4f}, Fingerprint Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct_fp, len(data_loader.dataset),
    #     100. * correct_fp / len(data_loader.dataset)))

    print('Fingerprints (on test): L(fp, y) loss: {:.4f}, L(fp, dy) loss: {:.4f}, argmax y = argmax f(x+dx) Accuracy: {}/{} ({:.0f}%)'.format(
        loss_y.data.cpu().numpy()[0], loss_dy.data.cpu().numpy()[0], num_same_argmax, len(data_loader.dataset) * args.num_dx,
        100. * argmax_acc))

    result = {"epoch": epoch,
              "test-loss": test_loss,
              "test-correct": correct,
              "test-N": test_length,
              "test-acc": correct/test_length,
              "fingerprint-loss (y)": loss_y.data[0],
              "fingerprint-loss (dy)": loss_dy.data[0],
              "fingerprint-loss (argmax)": argmax_acc,
              "args": args
              }
    path = os.path.join(args.log_dir, "train", "log-ep-{}.pkl".format(epoch))
    print("Saving log in", path)
    pickle.dump(result, open(path, "wb"))
    return test_loss
