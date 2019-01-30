import sys

# add the system path here!
# sys.path.append('')

import src.mister_ed.utils.pytorch_utils as utils
import argparse
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from outlier.model import PreActResNet18
from outlier.out_utils import get_dataset, evaluation, find_most_likely, build_adv_examples, odin_score

parser = argparse.ArgumentParser(description='CIFAR10 Outlier Detection Experiment')
parser.add_argument('--model_path', type=str, help='path to the model')
parser.add_argument('--is_fp', type=bool, default=True, help='Use fingerprint-trained model or normal model')
parser.add_argument('--in_name', type=str, default='cifar10', help='name of the in-distribution(ID) dataset')
parser.add_argument('--out_name', type=str, default='svhn', help='name of the ood dataset')
parser.add_argument('--num_class', type=str, default=10, help='number of classes')
parser.add_argument('--log_dir', type=str, help='path to the logging directory')
parser.add_argument('--size', type=int, default=10000, help='size of the testing set')
parser.add_argument('--num_dx', type=int, default=30, help='number of the fingerprint dx directions')
parser.add_argument('--cuda', type=bool, default=True, help="Use GPU or not")
parser.add_argument('--debug', type=bool, default=False, help="Use GPU or not")

args = parser.parse_args()

# parse those configurations
# Get in distribution dataset and OOD dataset
in_name = args.in_name
out_name = args.out_name
in_dataset = get_dataset(in_name)
out_dataset = get_dataset(out_name)
num_dx = args.num_dx
size = args.size

assert size > 1000  # we remove the first 1k images

num_class = args.num_class


torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)  # ignore this if using CPU

if in_name == 'cifar10':
    if args.is_fp:
        # use fp-trained model
        normalizer = utils.DifferentiableNormalize((0.5, 0.5, 0.5), (1, 1, 1))
    else:
        # use normal model
        normalizer = utils.DifferentiableNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

elif in_name == 'cifar100':
    normalizer = utils.DifferentiableNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

else:
    raise Exception("Unsupported In-Distribution Dataset!")

# get the model
classifier_net = PreActResNet18(num_class=args.num_class)

if args.cuda:
    classifier_net.cuda()

model_path = args.model_path

classifier_net.load_state_dict(torch.load(model_path))

# restore saved images
dis_in = []
dis_out = []

# Compute in distribution distance
for idx, in_data in enumerate(in_dataset, 0):
    # use batch size=1 currently
    if idx == size:
        break

    # ignore the first 1000 images
    if idx < 1000:
        continue

    in_image, _ = in_data

    if args.cuda:
        in_image = in_image.cuda()

    pred_in = find_most_likely(classifier_net, normalizer, in_image)

    # input pre-processing
    in_image = build_adv_examples(classifier_net, normalizer, in_image, pred_in,
                                  step_size=args.eps)

    with torch.no_grad():
        in_dis = odin_score(in_image, classifier_net, normalizer, args.temp_scale)
        dis_in.append(in_dis)

# Compute OOD distance
for idx, out_data in enumerate(out_dataset, 0):
    # use batch size=1 currently

    if idx == size:
        break

    # ignore the first 1000 images
    if idx < 1000:
        continue

    out_image, _ = out_data

    if args.cuda:
        out_image = out_image.cuda()

    pred_out = find_most_likely(classifier_net, normalizer, out_image)

    # input pre-processing
    out_image = build_adv_examples(classifier_net, normalizer, out_image, pred_out,
                                   step_size=args.eps)

    with torch.no_grad():
        out_dis = odin_score(out_image, classifier_net, normalizer, args.temp_scale)
        dis_out.append(out_dis)

# change to array
dis_in = np.asarray(dis_in)
dis_out = np.asarray(dis_out)

assert int(len(dis_in)) == size - 1000
assert int(len(dis_out)) == size - 1000

# compute some distance statistics
mean_in = np.mean(dis_in)
min_in = min(dis_in)
max_in = max(dis_in)
std_in = np.std(dis_in)

mean_out = np.mean(dis_out)
min_out = min(dis_out)
max_out = max(dis_out)
std_out = np.std(dis_out)

if args.debug:
    print("Successful Evaluation")
    print("MEAN IN", mean_in)
    print("MEAN OUT", mean_out)
    print("MAX IN", max_in)
    print("MAX OUT", max_out)
    print("STD IN", std_in)
    print("STD OUT", std_out)
    print("MIN IN", min_in)
    print("MIN OUT", min_out)

reject_thresholds = \
    [0.0000005 * i for i in range(0, 2000000)]

best_acc = 0
best = {}

# compute auroc
label_0 = np.zeros(size - 1000)
label_1 = np.ones(size - 1000)

labels = np.concatenate((label_0, label_1))
scores = np.concatenate((dis_out, dis_in))

auroc = roc_auc_score(labels, scores)

print("The AUROC IS ", auroc)

fpr = 0.
total = 0.

for tau in reject_thresholds:
    true_positive, false_positive, true_negative, false_negative = evaluation(dis_out, dis_in, tau)

    tpr = true_positive / np.float(true_positive + false_negative)

    error2 = false_positive / np.float(false_positive + true_negative)

    if 0.9505 >= tpr >= 0.9495:
        fpr += error2
        total += 1

    p_out_large = false_negative / (true_positive + false_negative)

    p_in_small = false_positive / (false_positive + true_negative)

    detection_acc = 1 - 1 / 2. * (p_out_large + p_in_small)

    if detection_acc > best_acc:
        best['tau'] = tau
        best['acc'] = detection_acc
        best['p_out_large'] = p_out_large
        best['p_in_small'] = p_in_small
        best_acc = detection_acc

# detection acc
print("The Detection Acc is ", best)
print("fpr ", fpr)
print("total ", total)

# fpr at 95 tpr
fprbase = 0
if total != 0:
    fprbase = fpr / total
else:
    raise Exception("Please Use Larger Interval or Finer Thresholds for Computing FPR at TPR 95")

print("The False Positive Rate is ", fprbase)
