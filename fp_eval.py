from __future__ import print_function
from collections import defaultdict
from PIL import Image
from sklearn.metrics import auc
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import codecs
import custom_datasets
import errno
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import os.path
import pickle
import random
import scipy as sp
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import util

from fingerprint import Example, Fingerprints, Stats

def find_nearest_neighbor(x):
    pass

def get_class_of(x):
    # return model(x)
    pass

def model_with_fingerprint(model, x, fp,  args):
    # x : B x C x W x H with B = 1
    # Check y' = f(x+dx) for all dx

    x = util.t2var(x, args.cuda)

    assert x.size()[0] == 1 # batch

    # Get perturbations for predicted class

    logits = model(x)
    log_yhat = F.log_softmax(logits)

    yhat = F.softmax(logits)
    y_class = yhat.data.max(1, keepdim=True)[1]
    y_class = util.t2np(y_class, args.cuda)[0,0]

    # fixed_dxs : num_perturb x C x W x H
    fixed_dxs = util.np2var(np.concatenate(fp.dxs, axis=0), cuda=args.cuda)

    # cmopute x + dx : broadcast! num_perturb x C x W x H
    xp = x + fixed_dxs

    # if args.debug: print("xp", xp.size(), "x", x.size(), "fixed_dxs", fixed_dxs.size())

    logits_p = model(xp)
    log_yhat_p  = F.log_softmax(logits_p)
    yhat_p = F.softmax(logits_p)

    if args.debug:
      print("logits_p", logits_p, "log_yhat_p", log_yhat_p)
      print("yhat_p", yhat_p)

    # compute f(x + dx) : num_perturb x num_class

    # print("get fixed_dys : num_target_class x num_perturb x num_class: for each target class, a set of perturbations and desired outputs (num_class).")
    fixed_dys = util.np2var(fp.dys, cuda=args.cuda)

    logits_norm = logits * torch.norm(logits, 2, 1, keepdim=True).reciprocal().expand(1, args.num_class)
    logits_p_norm = logits_p * torch.norm(logits_p, 2, 1, keepdim=True).reciprocal().expand(args.num_dx, args.num_class)

    if args.debug:
      print("logits_norm", logits_norm)
      print("logits_p_norm", logits_p_norm.size(), torch.norm(logits_p_norm, 2, 1))

    diff_logits_p = logits_p_norm - logits_norm
    #diff_logits_p = diff_logits_p * torch.norm(diff_logits_p, 2, 1, keepdim=True).reciprocal().expand(args.num_dx, args.num_class)


    diff = fixed_dys - diff_logits_p

    if args.debug:
      print("diff_logits_p", diff_logits_p)
      print("fixed_dys", fixed_dys)
      print("diff", diff)


    diff_norm = torch.norm(diff, 2, dim=2)

    if args.debug: print("diff_norm (over dim 2 of diff)", diff_norm)

    diff_norm = torch.mean(diff_norm, dim=1)

    if args.debug: print("diff_norm after mean", diff_norm)

    y_class_with_fp = diff_norm.data.min(0, keepdim=True)[1]
    y_class_with_fp = util.t2np(y_class_with_fp, args.cuda)[0]

    if args.debug:
      print("y_class_with_fp", y_class_with_fp, diff_norm.data.min(0, keepdim=True))

    ex = Example(x, yhat, y_class)
    ex.dxs = fixed_dxs
    ex.yhat_p = yhat_p
    ex.diff = diff
    ex.diff_norm = diff_norm
    ex.y_class_with_fp = y_class_with_fp

    return ex

def detect_with_fingerprints(ex, stats_per_tau, args):

    diff_norm = ex.diff_norm
    y_class_with_fp = ex.y_class_with_fp

    for reject_threshold, stats in stats_per_tau.items():

        stats.ids.add(ex.id)

        if ex.y == ex.y_class:
            stats.ids_correct.add(ex.id)

        if ex.y == ex.y_class_with_fp:
            stats.ids_correct_fp.add(ex.id)

        if ex.y_class == ex.y_class_with_fp:
            stats.ids_agree.add(ex.id)

        # Check legal: ? D({f(x+dx)}, {y^k}) < tau for all classes k.
        below_threshold = diff_norm < reject_threshold
        below_threshold_t = below_threshold[y_class_with_fp].data
        if args.cuda:
            below_threshold_t = below_threshold_t.cpu()
        is_legal = below_threshold_t.numpy()[0] > 0
        ex.is_legal = is_legal

        if ex.is_legal:
            stats.ids_legal.add(ex.id)

            if args.verbose:
                print("{} (D) < {:.2f} (tau) --> legal input".format(diff_norm.data.numpy()[y_class_with_fp], args.tau))
        else:
            # closest_x_in_data = find_nearest_neighbor(ex.x)
            # ex.y_class_with_fp = get_class_of(ex.x)

            if args.verbose:
                print("{} (D) >= {:.2f} (tau) --> illegal input".format(diff_norm.data.numpy()[y_class_with_fp], args.tau))

    return ex, stats_per_tau

def eval_with_fingerprints(model, data_loader, ds_name, fp, reject_thresholds, test_results_by_tau, args):

    stats_per_tau = {i: Stats(tau=i, name=args.name, ds_name=ds_name) for i in reject_thresholds}

    i = 0
    for e, (x,y) in enumerate(data_loader):
        data_np = x.numpy()
        real_bs = data_np.shape[0]
        for b in range(real_bs):

            ex = model_with_fingerprint(model, x[b:b+1], fp, args)

            # Careful! Needs Dataloader with shuffle=False
            ex.id = i

            ex.y = y[b]

            ex, stats_per_tau = detect_with_fingerprints(ex, stats_per_tau, args)

            i += 1

            if args.verbose:
                print("\nx", x[b:b+1].size(), "y", y[b:b+1], y[b:b+1].size())
                print("Fingerprinting image (hash:", hash(x[b:b+1]), ") class", y[b])
                print("Model    class prediction: [", ex.y_class, "] from logits:", ex.yhat)
                print("Model+fp class prediction: [{}] from diff_norm: {}".format(ex.y_class_with_fp, ex.diff_norm.data.numpy()))

        if e % 10 == 0:
            print("Ex: {} batch {} of size {}".format(i, e, real_bs))

        if args.debug and e >= 0:
            print("Debug break!")
            break


    if args.verbose: print("Stats for dataset:", args.data_dir)

    results = defaultdict(lambda:None)
    stats_results = defaultdict(lambda:None)

    for tau, stats in stats_per_tau.items():
        stats.counts = stats.compute_counts()

        # use test x for which f(x) was correct. If the current dataset is not test, we need an external set of ids.
        if ds_name == "test":
            ids_correct = stats.ids_correct
        elif test_results_by_tau:
            ids_correct = test_results_by_tau[tau].ids_correct
        else:
            continue

        stats.counts_correct = stats.compute_counts(ids_correct=ids_correct)

        if args.verbose:
            print("Stats raw (tau {})".format(tau))
            stats.show(stats.counts)
            print("Stats cond_correct (tau {})".format(tau))
            stats.show(stats.counts_correct)

        stats.dump(args)

        results[tau] = [stats.counts, stats.counts_correct]
        stats_results[tau] = stats

    return results,stats_results

def safe_div(a,b):
    if abs(b) < 1e-10:
        assert abs(a) < 1e-10
        return a*1.0 / (b + 1e-10)
    else:
        return a*1.0 / b

def get_rates(tau, args, results_by_dataset, pos_names=None, neg_names=None):
    # Get precision / recall
    # adversarial_data = random_loader # change this when adv work
    # pos_name = "random"

    # positive example = adversarial examples.
    # negative example = real image
    num_pos, num_neg, true_pos, false_neg, true_neg, false_pos = 0, 0, 0, 0, 0, 0

    for pos_name in pos_names:
        res = results_by_dataset[pos_name]

        num_pos += res["num"]
        true_pos += res["num_reject"]
        false_neg += res["num"] - res["num_reject"]

    for pos_name in neg_names:
        res = results_by_dataset[pos_name]

        num_neg += res["num"]
        true_neg += res["num"] - res["num_reject"]
        false_pos += res["num_reject"]

    prec = safe_div(true_pos, true_pos + false_pos)
    recall = safe_div(true_pos, true_pos + false_neg)

    tpr = safe_div(true_pos, true_pos + false_neg)
    fpr = safe_div(false_pos, false_pos + true_neg)

    if args.verbose:
        print("ROC + Precision/recall @ tau = {}".format(tau))
        print("{} positive ({}) {} negative ({}):".format(num_pos, pos_names, num_neg, neg_names))
        print("TP {} FP {} TN {} FN {} TPR {} FPR {} P {:1.2f} R {:1.2f}".format(true_pos, false_pos, true_neg, false_neg, tpr, fpr, prec, recall))
        print("Latex\n{} & {} & {} & {} & {:1.2f} & {:1.2f}".format(true_pos, false_pos, true_neg, false_neg, prec, recall))

    pr_result = {"true_pos": true_pos,
                "false_pos": false_pos,
                "true_neg": true_neg,
                "false_neg": false_neg,
                "tpr": tpr,
                "fpr": fpr,
                "prec": prec,
                "recall": recall}

    return pr_result

def get_pr_auc(pr_results, args, plot=False, plot_name=""):
    xys = [(0.0, 1.0, 0.)]
    labels = []
    for tau, result in pr_results.items():
        xys += [(pr_results[tau]["recall"], pr_results[tau]["prec"], tau)]
        labels += [tau]

    xys.sort(key=lambda x: x[0])
    xs = [i[0] for i in xys]
    ys = [i[1] for i in xys]

    # print("pr")
    # for i in sorted(xys, key=lambda x: x[-1]): print(i)

    # print("recall", xs)
    # print("precis", ys)
    _auc = auc(xs, ys)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xs, ys, 'go-',)

        for label, x, y in zip(labels, xs, ys):
            ax.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        path = os.path.join(args.log_dir, "pr-{}.svg".format(plot_name))
        print("Storing PR plot in", path)
        fig.savefig(path)
        plt.close(fig)

    return _auc

def get_roc_auc(pr_results, args, plot=False, plot_name=""):
    xys = [(0.0, 0.0, 0.0)]
    labels = []
    for tau, result in pr_results.items():
        xys += [(pr_results[tau]["fpr"], pr_results[tau]["tpr"], tau)]
        labels += [tau]

    xys.sort(key=lambda x: x[0])
    xs = [i[0] for i in xys]
    ys = [i[1] for i in xys]

    # print("roc")
    # for i in sorted(xys, key=lambda x: x[-1]): print(i)

    # print("fpr", xs)
    # print("tpr", ys)
    _auc = auc(xs, ys)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(xs, ys, 'go-',)

        for label, x, y in zip(labels, xs, ys):
            ax.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        path = os.path.join(args.log_dir, "roc-{}.svg".format(plot_name))
        print("Storing ROC plot in", path)
        fig.savefig(path)
        plt.close(fig)

    return _auc

def get_pr_wrapper(results, pos_names, neg_names, reject_thresholds, args):

    for e, _type in enumerate(["raw", "cond_correct"]):

        pr_results = {}

        for tau in reject_thresholds:

            tmp_results = {k: v[tau][e] for k,v in results.items()}

            rates = get_rates(tau, args, tmp_results,
                pos_names=pos_names, neg_names=neg_names)

            if not (rates["prec"] < 1e-10 and rates["recall"] < 1e-10):
                pr_results[tau] = rates

        pr_auc = get_pr_auc(pr_results, args, plot=True, plot_name="{}-{}-{}".format(_type, "-".join(pos_names), "-".join(neg_names)))
        roc_auc = get_roc_auc(pr_results, args, plot=True, plot_name="{}-{}-{}".format(_type, "-".join(pos_names), "-".join(neg_names)))
        print(pos_names, neg_names, _type, "{}: AUC ROC {} PR {}".format(_type, roc_auc, pr_auc))

        pr_results["pr_auc"] = pr_auc
        pr_results["roc_auc"] = roc_auc

        # print("count stats")
        # for k,v in pr_results.items():
        #     print(k,v)

        path = os.path.join(args.log_dir, "rates-roc-pr-auc_{}_{}_{}_tau_{:.4f}.pkl".format(_type, "-".join(pos_names), "-".join(neg_names), tau))
        print("Saving pr result in", path)
        pickle.dump(pr_results, open(path, "wb"))
