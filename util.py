from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from src.mister_ed.loss_functions import PartialXentropy
import os
import keras
from keras import backend as K
import numpy as np
import dill as pickle
import outlier.data_loader as dl
import numpy as np
import os
import src.mister_ed.utils.pytorch_utils as utils
import src.mister_ed.adversarial_attacks as aa
import src.mister_ed.adversarial_perturbations as ap
import torch


def test_tf2torch(tf_model, torch_model, input_shape, num_rand_inp=10, precision=10 ** -2):
    """
    Checks consistency of torch and tf models before generating attacks
    :param tf_model: copied tf model
    :param torch_model: torch model to be transferred to tf
    :param input_shape: Format Channels X Height X Width
    :param num_rand_inp: number of random inputs to test consistency on
    :return: raises error if the outputs are not consistent
    """
    torch_model.eval()
    rand_x = torch.rand(num_rand_inp, input_shape[0], input_shape[1], input_shape[2])
    tf_op = tf_model.predict(rand_x.numpy())
    torch_op = F.softmax(torch_model(Variable(rand_x))).data.numpy()
    assert tf_op.shape == torch_op.shape, "Mismatch of dimensions of the outputs from tf and torch models"
    assert np.linalg.norm(torch_op - tf_op) / np.linalg.norm(
        torch_op) <= num_rand_inp * precision, "Outputs of the torch and tensorflow models" \
                                               "do not agree"
    pass


def np2var(x, cuda):
    if cuda:
        return Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor))
    else:
        return Variable(torch.from_numpy(x).type(torch.FloatTensor))


def var2np(x, cuda):
    t = x.data
    if cuda:
        t = t.cpu()
    return t.numpy()


def t2var(t, cuda):
    if cuda:
        t = t.cuda()
    t = Variable(t, volatile=True)
    return t


def t2np(t, cuda):
    if cuda:
        t = t.cpu()
    return t.numpy()


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def add_boolean_argument(parser, name, default=False):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')


def torch2tf(torch_Net, tf_Net):
    return tf_Net


def get_dataset(name):
    """"Return a dataloader of the specified dataset"""
    print(name)
    if name == "cifar10":
        dataset = dl.load_cifar10_data('val', shuffle=False, batch_size=1)
    elif name == "cifar100":
        dataset = dl.load_cifar100_data('val', shuffle=False, batch_size=1)
    elif name == "svhn":
        dataset = dl.load_svhn_data('val', shuffle=False, batch_size=1)
    elif name == 'lsun':
        dataset = dl.load_lsun_data(shuffle=False, batch_size=1)
    elif name == 'imagenet':
        dataset = dl.load_imagenet_data(shuffle=False, batch_size=1)
    elif name == 'place':
        dataset = dl.load_place365_data(shuffle=False, batch_size=1)
    else:
        raise Exception("Unsupported Dataset!")

    return dataset


def get_finger_print(fp_dir, is_cuda):
    """restore fingerprints"""
    fingerprint_dir = fp_dir

    fixed_dxs = np.load(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), encoding='bytes')
    fixed_dys = np.load(os.path.join(fingerprint_dir, "fp_outputs.pkl"), encoding='bytes')

    # preprocessing
    fixed_dxs = utils.np2var(np.concatenate(fixed_dxs, axis=0), is_cuda=is_cuda)
    fixed_dys = utils.np2var(fixed_dys, is_cuda=is_cuda)

    return fixed_dxs, fixed_dys


def ood_evaluation(dis_ood, dis_real, tau):
    """Compute the OOD stats given two distance array(ID and OOD) and the threshold"""
    true_positive = np.sum(dis_real >= tau)
    false_negative = np.sum(dis_real < tau)

    false_positive = np.sum(dis_ood >= tau)
    true_negative = np.sum(dis_ood < tau)

    return true_positive, false_positive, true_negative, false_negative


def adv_evaluation(dis_adv, dis_real, tau):
    """Compute the statistics of NFP distance on adversarial images and real images.
    FP distance on adv images is normally much larger than the respective distance
    on real distance."""
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for adv in dis_adv:
        if adv > tau:
            true_positive += 1
        else:
            false_negative += 1

    for real in dis_real:
        if real > tau:
            false_positive += 1
        else:
            true_negative += 1

    return true_positive, false_positive, true_negative, false_negative


def build_fgsm_adv_examples(classifier_net, normalizer, inputs, labels, step_size):
    """Program to build adversarial example with the given images and labels. Here we do pre-processing
    of the image using FGSM(Fast Gradient Sign Method)."""
    threat_model = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                     'lp_bound': 1})
    attack_loss = PartialXentropy(classifier_net, normalizer=normalizer)

    attack_object = aa.FGSM(classifier_net, normalizer, threat_model, attack_loss)

    perturbation_out = attack_object.attack(inputs, labels, step_size=step_size, verbose=False)
    adv_examples = perturbation_out.adversarial_tensors()

    return adv_examples


def odin_score(adv_inputs, net, normalizer, temper):
    """Compute the ODIN score with temperature scaling. Note: the inputs should be perturbed using the build
    _adv_example method"""
    with torch.no_grad():
        outputs = net(normalizer(adv_inputs))

        # temperature scaling
        outputs = outputs / temper
        nnOutputs = outputs.cpu().numpy()

        # minus maximum value to avoid overflow
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

    return np.max(nnOutputs)


def softmax_score(inputs, net, normalizer):
    """"Compute the maximum value of the model's softmax output"""
    with torch.no_grad():
        outputs = net(normalizer(inputs))
        nnOutputs = outputs.cpu().numpy()

        # minus the maximum value to avoid overflow
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

    return np.max(nnOutputs)


def find_most_likely(classifier_net, normalizer, image):
    """Find the most likely class of the image"""
    logits = classifier_net.forward(normalizer.forward(image))
    indice = torch.argmax(logits, dim=1)

    return indice
