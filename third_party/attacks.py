from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_first')
from keras.models import load_model
import dill as pickle
from third_party.lid_adversarial_subspace_detection.util import (get_data, get_model, cross_entropy, get_noisy_samples)
from third_party.lid_adversarial_subspace_detection.attacks import (fast_gradient_sign_method, basic_iterative_method,
                         saliency_map_method)
from third_party.lid_adversarial_subspace_detection.cw_attacks import CarliniL2, CarliniLID

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 32, 'num_channels': 3, 'num_labels': 10}
}

# CLIP_MIN = 0.0
# CLIP_MAX = 1.0
# PATH_DATA = "../data/"

CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "./adv_examples/"

def craft_one_type(sess, model, X, Y, dataset, attack, batch_size, log_path=None):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """
    if not log_path is None:
        PATH_DATA = log_path

    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size
        )
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take > 5 hours')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=CLIP_MIN, clip_max=CLIP_MAX
        )
    elif attack == 'cw-l2':
        # C&W attack
        print('Crafting %s examples. This takes > 5 hours due to internal grid search' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniL2(sess, model, image_size, num_channels, num_labels, batch_size=batch_size)
        X_adv = cw_attack.attack(X, Y)
    elif attack == 'cw-lid':
        # C&W attack to break LID detector
        print('Crafting %s examples. This takes > 5 hours due to internal grid search' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniLID(sess, model, image_size, num_channels, num_labels, batch_size=batch_size)
        X_adv = cw_attack.attack(X, Y)

    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100.0 * acc))
    _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))

    #np.save(os.path.join(PATH_DATA, 'Adv_%s_%s.npy' % (dataset, attack)), X_adv)
    f = open(os.path.join(log_path,'Adv_%s_%s.p' % (dataset, attack)),'w')

    pickle.dump({"adv_input":X_adv,"adv_labels":Y},f)
    f.close()

    X_nsy = get_noisy_samples(X, X_adv, dataset, attack)
    _, acc = model.evaluate(X_nsy, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))

    f = open(os.path.join(log_path,'Noisy_%s_%s.p' % (dataset, attack)),'w')
    pickle.dump({"adv_input":X_nsy,"adv_labels":Y},f)
    f.close()

    l2_diff = np.linalg.norm(
        X_adv.reshape((len(X), -1)) -
        X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print("Average L-2 perturbation size of the %s attack: %0.2f" %
          (attack, l2_diff))
    return (X_adv,Y)

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'all', 'cw-lid'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2', 'all' or 'cw-lid' for attacking LID detector"
    model_file = os.path.join(PATH_DATA, "model_%s.h5" % args.dataset)
    # model_file = "../data_v1/model_%s.h5" % args.dataset
    print(model_file)
    assert os.path.isfile(model_file), \
        'model file not found... must first train model using train_model.py.'
    if args.dataset == 'svhn' and args.attack == 'cw-l2':
        assert args.batch_size == 16, \
        "svhn has 26032 test images, the batch_size for cw-l2 attack should be 16, " \
        "otherwise, there will be error at the last batch!"


    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    if args.attack == 'cw-l2' or args.attack == 'cw-lid':
        warnings.warn("Important: remove the softmax layer for cw attacks!")
        # use softmax=False to load without softmax layer
        model = get_model(args.dataset, softmax=False)
        model.compile(
            loss=cross_entropy,
            optimizer='adadelta',
            metrics=['accuracy']
        )
        model.load_weights(model_file)
    else:
        model = load_model(model_file)

    _, _, X_test, Y_test = get_data(args.dataset)
    _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size,
                            verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))

    if args.attack == 'cw-lid': # breaking LID detector - test
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]

    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm']:
            craft_one_type(sess, model, X_test, Y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(sess, model, X_test, Y_test, args.dataset, args.attack,
                       args.batch_size)
    print('Adversarial samples crafted and saved to %s ' % PATH_DATA)
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', or 'cw-l2' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=128)
    # args = parser.parse_args()
    for dataset in ['mnist']:
        args = parser.parse_args(['-d', dataset, '-a', 'fgsm', '-b', '100'])
        main(args)

    # cifar
    # args = parser.parse_args(['-d', 'cifar', '-a', 'cw-lid', '-b', '100'])
    # main(args)
    #
    # # svhn
    # args = parser.parse_args(['-d', 'svhn', '-a', 'cw-lid', '-b', '16'])
    # main(args)