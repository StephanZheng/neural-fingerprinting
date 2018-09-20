from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_first')
from keras.models import load_model
import dill as pickle
from third_party.lid_adversarial_subspace_detection.util import (get_data, get_model, cross_entropy) # , get_noisy_samples)
from third_party.lid_adversarial_subspace_detection.attacks \
            import (fast_gradient_sign_method, basic_iterative_method,
                                          saliency_map_method)

from third_party.lid_adversarial_subspace_detection.adaptive_attacks \
            import adaptive_fast_gradient_sign_method

from third_party.lid_adversarial_subspace_detection.adaptive_attacks \
            import adaptive_basic_iterative_method

from third_party.lid_adversarial_subspace_detection.cw_attacks import CarliniL2, CarliniFP, CarliniFP_2vars
from cleverhans.attacks import SPSA

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 32, 'num_channels': 3, 'num_labels': 10}
}

# CLIP_MIN = 0.0
# CLIP_MAX = 1.0
# PATH_DATA = "../data/"

from cleverhans.model import Model, CallableModelWrapper


CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "./adv_examples/"

def craft_one_type(sess, model, X, Y, dataset, attack, batch_size, log_path=None,
                   fp_path = None, model_logits = None):
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
    elif attack == 'adapt-fgsm':
        # Adaptive FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = adaptive_fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size,
            log_dir = fp_path,
            model_logits = model_logits,
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
    elif attack == 'cw-fp':
        # C&W attack to break LID detector
        print('Crafting %s examples. This takes > 5 hours due to internal grid search' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniFP_2vars(sess, model, image_size, num_channels, num_labels, batch_size=batch_size,
                              fp_dir=fp_path)
        X_adv = cw_attack.attack(X, Y)

    elif attack == 'spsa':
        print('Crafting %s examples. Using Cleverhans' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']

        from cleverhans.utils_keras import KerasModelWrapper
        wrapped_model = KerasModelWrapper(model)

        if dataset == "mnist": 
            wrapped_model.nb_classes = 10 
        elif dataset == "cifar": 
            wrapped_model.nb_classes = 10 
        else:
            wrapped_model.nb_classes = 10 

        real_batch_size = X.shape[0]

        spsa = SPSA(wrapped_model, back='tf', sess=sess)
        spsa_params = {
            "epsilon": 4. / 255,
            'num_steps': 1,
            'spsa_iters': 1,
            'early_stop_loss_threshold': -1.,
            'is_targeted': False,
            'is_debug': True,
            'spsa_samples': real_batch_size,
        }   
        batch_shape = X.shape
        X_input = tf.placeholder(tf.float32, shape=(1,) + X.shape[1:])
        Y_label = tf.placeholder(tf.float32, shape=(1,) + Y.shape[1:])
        print("log_dir", fp_path)
        X_adv_spsa = spsa.generate(X_input, y=Y_label, log_dir=fp_path, **spsa_params)
    
        # X = (X - np.argmin(X))/(np.argmax(X)-np.argmin(X))
        X_adv = []
        for i in range(real_batch_size):        
        
            # rescale to format TF wants
            _min = np.min(X[i])
            _max = np.max(X[i])
            X_i_norm = (X[i] - _min)/(_max-_min)
       
            # Run attack
            res = sess.run(X_adv_spsa, feed_dict={X_input: np.expand_dims(X_i_norm, axis=0), Y_label: np.array([np.argmax(Y[i])])})
        
            # Rescale result back to our scale
            X_adv += [(res + _min) * (_max-_min)]
            
        X_adv = np.concatenate(X_adv, axis=0)
        # X_adv = spsa.generate_np(X, **spsa_params) 

    print(X.shape, X_adv.shape, Y.shape)

    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100.0 * acc))
    _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))

    if("adapt" in attack or "fp" in attack):
        [m,_,_,_]=(np.shape(X_adv))
        cropped_X_adv = []
        cropped_Y = []
        cropped_X = []
        if(dataset == 'mnist'):
            X_place = tf.placeholder(tf.float32, shape=[1, 1, 28, 28])
            pred = model(X_place)
        print(m)
        for i in range(m):
            logits_op = sess.run(pred,feed_dict={X_place:X_adv[i:i+1,:,:,:],
                                           K.learning_phase(): 0})
            if(not np.argmax(logits_op) == np.argmax(Y[i,:])):
                cropped_Y.append(Y[i,:])
                cropped_X_adv.append(X_adv[i,:,:,:])
                cropped_X.append(X[i,:,:,:])
        X_adv = np.array(cropped_X_adv)
        X = np.array(cropped_X)
        Y = np.array(cropped_Y)

    print(len(X_adv))

    #np.save(os.path.join(PATH_DATA, 'Adv_%s_%s.npy' % (dataset, attack)), X_adv)
    f = open(os.path.join(log_path,'Adv_%s_%s.p' % (dataset, attack)),'w')

    pickle.dump({"adv_input":X_adv,"adv_labels":Y},f)
    f.close()

    print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))

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

