import outlier.data_loader as dl
import numpy as np
import os
import src.mister_ed.utils.pytorch_utils as utils
import src.mister_ed.adversarial_attacks as aa
import src.mister_ed.adversarial_perturbations as ap
import torch

from src.mister_ed.loss_functions import PartialXentropy


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


def evaluation(dis_ood, dis_real, tau):
    """Compute the OOD stats given two distance array(ID and OOD) and the threshold"""
    true_positive = np.sum(dis_real >= tau)
    false_negative = np.sum(dis_real < tau)

    false_positive = np.sum(dis_ood >= tau)
    true_negative = np.sum(dis_ood < tau)

    return true_positive, false_positive, true_negative, false_negative


def build_adv_examples(classifier_net, normalizer, inputs, labels, step_size):
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
