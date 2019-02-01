from __future__ import print_function

import sys

# change this to your system path
# sys.path.append('')
from sklearn.metrics import roc_auc_score

from src.mister_ed.loss_functions import AdversarialRegularizedLoss
from cifar.model import CW2_Net
from src.mister_ed.loss_functions import CWLossF6
from util import get_finger_print
from src.mister_ed.loss_functions import NFLoss2, NFLoss
from white_box.data_loader import load_cifar10_data
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import src.mister_ed.utils.pytorch_utils as utils
import src.mister_ed.adversarial_perturbations as ap
import src.mister_ed.adversarial_attacks as aa

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--model_path', type=str, help='path to the fingerprint trained model')
parser.add_argument('--fp_dir', type=str, help='path to the fingerprint directory')
parser.add_argument('--size', type=int, default=100, help='size of the testing set')
parser.add_argument('--num_iterations', type=int, default=50000, help='attack iteration steps')
parser.add_argument('--lp_bound', type=float, default=8.0 / 255.0, help='eps for the norm ball')
parser.add_argument('--step_size', type=float, default=0.00001, help='attack step size')
parser.add_argument('--cuda', type=bool, default=True, help="Use GPU or not")

args = parser.parse_args()


def build_adv_examples(perturb, classifier_net, normalizer, threat_model, attack_loss, num_iterations, length,
                       inputs, labels):
    """Program to build adversarial example with the given images and labels"""

    attack_object = perturb(classifier_net, normalizer, threat_model, attack_loss)
    perturbation_out = attack_object.attack(inputs, labels, num_iterations=num_iterations, step_size=length,
                                            random_init=False, keep_best=False, verbose=False, signed=True,
                                            use_momentum=True, momentum=0.9)
    adv_examples = perturbation_out.adversarial_tensors()
    return adv_examples


def build_attack_loss(classifier_net, normalizer, loss, relative_weight, l_real, target):
    vanilla_loss = CWLossF6(classifier_net, normalizer)
    losses = {'vanilla': vanilla_loss, 'fingerprint': loss}
    scalars = {'vanilla': -1., 'fingerprint': -1. * relative_weight}

    combine_loss = AdversarialRegularizedLoss(losses=losses, scalars=scalars, l_real=l_real, target=target)

    return combine_loss


def get_prediction(normalizer, examples, classifier_net):
    """Find the label of current image"""
    if normalizer is not None:
        examples = normalizer.forward(examples)

    logits = classifier_net.forward(examples)
    yhat = F.softmax(logits, dim=1)
    pred = yhat.data.max(1, keepdim=True)[1]

    return pred


def find_second_likely(classifier_net, normalizer, input):
    # compute logits
    logits = classifier_net.forward(normalizer.forward(input))

    # pick the second largest label

    # output is a list with 2 elements
    _, indices = torch.topk(logits, 2, dim=1)

    # get the second likely label
    output = indices[0][1]

    return output


def build_advs(dataset, num_iterations, threat_model, attack_method, weight, step_size):
    """
    :param step_size: Attack Step Size. Should set to really small value
    :param dataset: default: cifar10
    :param num_iterations: Iteration Steps for gradient-based Attacks
    :param threat_model: Attack threatmodel
    :param attack_method: Attack Method. PGD/CW2
    :param weight: relative importance between detection performance and fingerprint matching loss
    """
    torch.manual_seed(1)

    if args.cuda:
        torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # get the model
    classifier_net = CW2_Net()

    # load the weight
    path = args.model_path

    classifier_net.load_state_dict(torch.load(path))

    if args.cuda:
        classifier_net.cuda()
    classifier_net.eval()

    fixed_dxs, fixed_dys = get_finger_print(args.fp_dir)

    # build loss object
    loss = NFLoss2(classifier_net, num_dx=30, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys,
                   normalizer=normalizer)

    # list that contains the respective real images of the adversarial images
    # we only need these two files for evaluation
    real_images = []
    adv_images = []

    correct_images = []  # contains all correctly classified images
    misclassify_images = []

    image_labels = []  # list of labels

    for idx, test_data in enumerate(dataset, 0):
        if idx is args.size:
            break

        inputs, labels = test_data

        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # find the second most likely label to target
        target = find_second_likely(classifier_net, normalizer, inputs)

        # compute real image fp loss
        l_real = loss.forward(inputs.detach(), labels.detach())
        loss.zero_grad()

        # build up attack loss
        attack_loss = build_attack_loss(classifier_net, normalizer, loss, weight, l_real.detach(), target)

        # build adversarial example
        adv_examples = build_adv_examples(attack_method, classifier_net, normalizer, threat_model, attack_loss,
                                          num_iterations, step_size, inputs, labels)
        assert adv_examples.size(0) is 1

        loss.zero_grad()

        # get the label of real and adversarial images
        adv_class = get_prediction(normalizer, adv_examples.detach(), classifier_net)
        real_class = get_prediction(normalizer, inputs.detach(), classifier_net)

        if labels.cpu().numpy() != real_class.cpu().numpy():
            print("Misclassify!")
            misclassify_images.append(inputs.cpu().detach().numpy())

        if labels.cpu().numpy() == real_class.cpu().numpy():
            print("Success Classify")
            correct_images.append(inputs.cpu().detach().numpy())

        if labels.cpu().numpy() != adv_class.cpu().numpy() and labels.cpu().numpy() == real_class.cpu().numpy():
            print("Success Attack")
            real_images.append(inputs.cpu().detach().numpy())
            adv_images.append(adv_examples.cpu().detach().numpy())

        image_labels.append(labels.cpu().numpy())

    log_path = args.log_dir

    np.save(os.path.join(log_path, "correct_real_images.npy"), correct_images)
    np.save(os.path.join(log_path, "misclassify_real_images.npy"), misclassify_images)
    np.save(os.path.join(log_path, "adv_images.npy"), adv_images)
    np.save(os.path.join(log_path, "real_images.npy"), real_images)
    np.save(os.path.join(log_path, "labels.npy"), image_labels)


def test_advs():
    """
    compute the AUROC statistics
    """
    torch.manual_seed(1)

    if args.cuda:
        torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # get the model
    classifier_net = CW2_Net()

    path = args.model_path

    classifier_net.load_state_dict(torch.load(path))

    if args.cuda:
        classifier_net.cuda()
    classifier_net.eval()

    fixed_dxs, fixed_dys = get_finger_print(args.fp_dir)

    # build loss object
    loss = NFLoss(classifier_net, num_dx=30, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys,
                  normalizer=normalizer)

    # restore saved images
    adv_image_path = os.path.join(args.log_dir, "adv_images.npy")
    real_image_path = os.path.join(args.log_dir, "real_images.npy")

    advs = np.load(adv_image_path, encoding='bytes')
    reals = np.load(real_image_path, encoding='bytes')

    dis_real = []
    dis_adv = []

    num_images = int(len(advs))

    for i in range(num_images):
        real = utils.np2var(reals[i], is_cuda=True)
        adv = utils.np2var(advs[i], is_cuda=True)

        l_real = loss.forward(real, labels=None)
        dis_real.append(l_real.cpu().detach().numpy())

        l_adv = loss.forward(adv, labels=None)
        dis_adv.append(l_adv.cpu().detach().numpy())

    mean_adv = np.mean(dis_adv)
    min_adv = min(dis_adv)
    max_adv = max(dis_adv)
    std_adv = np.std(dis_adv)

    # because sometimes not all images are adverarial, we compute real distance wrt all correctly classified images
    mean_real = np.mean(dis_real)
    min_real = min(dis_real)
    max_real = max(dis_real)
    std_real = np.std(dis_real)

    print("MEAN REAL", mean_real)
    print("MEAN ADV", mean_adv)
    print("MAX REAL", max_real)
    print("MAX ADV", max_adv)
    print("MIN REAL", min_real)
    print("MIN ADV", min_adv)
    print("STD REAL", std_real)
    print("STD ADV", std_adv)

    label_0 = np.zeros(num_images)
    label_1 = np.ones(num_images)

    labels = np.concatenate((label_0, label_1))
    scores = np.concatenate((dis_real, dis_adv))

    auroc = roc_auc_score(labels, scores)

    print("AUROC is ", auroc)


if __name__ == '__main__':
    # currently not using the custom_datset
    test_dataloader = load_cifar10_data('val', shuffle=False, batch_size=1)
    num_iter = args.num_iterations
    threat_model = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                     'lp_bound': args.lp_bound})
    gamma = 1
    attack = aa.PGD
    step_length = args.step_size

    build_advs(dataset=test_dataloader, num_iterations=num_iter, threat_model=threat_model,
               attack_method=attack, weight=gamma, step_size=step_length)

    test_advs()
