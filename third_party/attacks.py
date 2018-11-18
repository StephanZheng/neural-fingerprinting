from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
K.set_image_data_format('channels_first')
from keras.models import load_model
import tensorflow.contrib.layers as layers
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
from cleverhans.attacks import MadryEtAl
keras.layers.core.K.set_learning_phase(0)

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
    print("entered")
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
            model_logits = model_logits, dataset = dataset
        )
    elif attack == 'adapt-bim-b':
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        X_adv = adaptive_basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size,
            log_dir = fp_path,
            model_logits = model_logits,  dataset = dataset)
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
        binary_steps = 1
        batch_shape = X.shape
        X_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
        Y_label = tf.placeholder(tf.int32, shape=(1,))
        alpha = tf.placeholder(tf.float32, shape= (1,))

        num_samples = np.shape(X)[0]
        # X = (X - np.argmin(X))/(np.argmax(X)-np.argmin(X))
        _min = np.min(X)
        _max = np.max(X)
        print(_max, _min)
        print(tf.trainable_variables())
        filters = sess.run('conv1/kernel:0')
        biases = 0.0*sess.run('conv1/bias:0')
        shift_model = Sequential()
        if(dataset == 'mnist'):
            shift_model.add(Conv2D(32, kernel_size=(3, 3),
                             activation=None,
                             input_shape=(1, 28, 28)))
        else:
            shift_model.add(Conv2D(32, kernel_size=(3, 3),
                             activation=None,
                             input_shape=(3, 32, 32)))

        X_input_2 = tf.placeholder(tf.float32, shape=(None,) + batch_shape[1:])

        correction_term = shift_model(X_input_2)
        if(dataset == 'mnist'):
            X_correction = -0.5*np.ones((1,1,28,28)) # We will shift the image up by 0.5, so this is the correction
        else:
            X_correction = -0.5*np.ones((1,3,32,32)) # We will shift the image up by 0.5, so this is the correction

        # for PGD


        shift_model.layers[0].set_weights([filters,biases])
        bias_correction_terms =(sess.run(correction_term, feed_dict={X_input_2:X_correction}))
        for i in range(32):
            biases[i] = bias_correction_terms[0,i,0,0]
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))
        original_biases = model.layers[0].get_weights()[1]
        original_weights = model.layers[0].get_weights()[0]
        model.layers[0].set_weights([original_weights,original_biases+biases])
        #Correct model for input shift

        X = X + 0.5 #shift input to make it >=0
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))
        # check accuracy post correction of input and model
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
        X_adv = None

        spsa = SPSA(wrapped_model, back='tf', sess=sess)
        spsa_params = {
            "epsilon": ATTACK_PARAMS[dataset]['eps'],
            'num_steps': 100,
            'spsa_iters': 1,
            'early_stop_loss_threshold': None,
            'is_targeted': False,
            'is_debug': False,
            'spsa_samples': real_batch_size,
        }
        X_adv_spsa = spsa.generate(X_input, alpha=alpha, y=Y_label, fp_path=fp_path, **spsa_params)

        for i in range(num_samples):

            # rescale to format TF wants

            #X_i_norm = (X[i] - _min)/(_max-_min)

            X_i_norm = X[i]
            # Run attack
            best_res = None
            ALPHA = np.ones(1)*0.1
            lb = 1.0e-2
            ub = 1.0e2
            for j in range(binary_steps):
                res = sess.run(X_adv_spsa,
                feed_dict={X_input: np.expand_dims(X_i_norm, axis=0), Y_label: np.array([np.argmax(Y[i])]),
                            alpha: ALPHA})
                if(dataset == 'mnist'):
                    X_place = tf.placeholder(tf.float32, shape=[1, 1, 28, 28])
                else:
                    X_place = tf.placeholder(tf.float32, shape=[1, 3, 32, 32])
                pred = model(X_place)
                model_op = sess.run(pred,feed_dict={X_place:res
                                               })

                if(not np.argmax(model_op) == np.argmax(Y[i,:])):
                    lb = ALPHA[0]
                else:
                    ub = ALPHA[0]
                ALPHA[0] = 0.5*(lb+ub)
                print(ALPHA)
                if(best_res is None):
                    best_res = res
                else:
                    if(not np.argmax(model_op) == np.argmax(Y[i,:])):
                        best_res = res
                        pass

            # Rescale result back to our scale

            if(i==0):
                X_adv = best_res
            else:
                X_adv = np.concatenate((X_adv,best_res), axis=0)





        _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the adversarial test set: %0.2f%%" % (100.0 * acc))
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))


        #Revert model to original
        model.layers[0].set_weights([original_weights,original_biases])
        #Revert adv shift
        X_adv = X_adv - 0.5

    elif attack=='adapt-pgd':
        binary_steps = 1
        rand_starts = 2
        batch_shape = X.shape
        X_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
        Y_label = tf.placeholder(tf.int32, shape=(1,))
        alpha = tf.placeholder(tf.float32, shape= (1,))

        num_samples = np.shape(X)[0]
        # X = (X - np.argmin(X))/(np.argmax(X)-np.argmin(X))
        _min = np.min(X)
        _max = np.max(X)
        print(_max, _min)
        print(tf.trainable_variables())
        filters = sess.run('conv1/kernel:0')
        biases = 0.0*sess.run('conv1/bias:0')
        shift_model = Sequential()
        if(dataset == 'mnist'):
            shift_model.add(Conv2D(32, kernel_size=(3, 3),
                             activation=None,
                             input_shape=(1, 28, 28)))
        else:
            shift_model.add(Conv2D(32, kernel_size=(3, 3),
                             activation=None,
                             input_shape=(3, 32, 32)))

        X_input_2 = tf.placeholder(tf.float32, shape=(None,) + batch_shape[1:])

        correction_term = shift_model(X_input_2)
        if(dataset == 'mnist'):
            X_correction = -0.5*np.ones((1,1,28,28)) # We will shift the image up by 0.5, so this is the correction
        else:
            X_correction = -0.5*np.ones((1,3,32,32)) # We will shift the image up by 0.5, so this is the correction

        # for PGD


        shift_model.layers[0].set_weights([filters,biases])
        bias_correction_terms =(sess.run(correction_term, feed_dict={X_input_2:X_correction}))
        for i in range(32):
            biases[i] = bias_correction_terms[0,i,0,0]
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))
        original_biases = model.layers[0].get_weights()[1]
        original_weights = model.layers[0].get_weights()[0]
        model.layers[0].set_weights([original_weights,original_biases+biases])
        #Correct model for input shift

        X = X + 0.5 #shift input to make it >=0
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))
        # check accuracy post correction of input and model
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
        X_adv = None

        pgd = MadryEtAl(wrapped_model, back='tf', sess=sess)
        X_adv_pgd, adv_loss_fp = pgd.generate(X_input, eps=0.3, eps_iter=0.02,
                                      clip_min=0.0, clip_max=1.0,
                        nb_iter=20, rand_init=True, fp_path=fp_path,
                        alpha=alpha)



        for i in range(num_samples):
            # rescale to format TF wants

            #X_i_norm = (X[i] - _min)/(_max-_min)

            X_i_norm = X[i]
            # Run attack
            best_res = None
            best_res_loss = 1000000.0
            ALPHA = np.ones(1)*0.1
            lb = 1.0e-2
            ub = 1.0e2
            for j in range(binary_steps):
                bin_flag = 0
                for jj in range(rand_starts):

                    [res,res_loss] = sess.run([X_adv_pgd, adv_loss_fp],
                        feed_dict={X_input: np.expand_dims(X[i], axis=0),
                        Y_label: np.array([np.argmax(Y[i])]),
                         alpha: ALPHA})

                    if(dataset == 'mnist'):
                        X_place = tf.placeholder(tf.float32, shape=[1, 1, 28, 28])
                    else:
                        X_place = tf.placeholder(tf.float32, shape=[1, 3, 32, 32])

                    pred = model(X_place)
                    model_op = sess.run(pred,feed_dict={X_place:res})


                    if(best_res is None):
                        best_res = res
                    else:
                        if((not np.argmax(model_op) == np.argmax(Y[i,:]))
                        and res_loss < best_res_loss):
                            best_res = res
                            best_res_loss = res_loss
                            bin_flag = 1
                            pass
                if(bin_flag == 1):
                    lb = ALPHA[0]
                else:
                    ub = ALPHA[0]
                ALPHA[0] = 0.5*(lb+ub)
                print(ALPHA)
            # Rescale result back to our scale

            if(i==0):
                X_adv = best_res
            else:
                X_adv = np.concatenate((X_adv,best_res), axis=0)





        _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the adversarial test set: %0.2f%%" % (100.0 * acc))
        _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))


        #Revert model to original
        model.layers[0].set_weights([original_weights,original_biases])
        #Revert adv shift
        X_adv = X_adv - 0.5
        pass

    if("adapt" in attack or "fp" in attack or "spsa" in attack):
        [m,_,_,_]=(np.shape(X_adv))
        cropped_X_adv = []
        cropped_Y = []
        cropped_X = []
        if(dataset == 'mnist'):
            X_place = tf.placeholder(tf.float32, shape=[1, 1, 28, 28])
            pred = model(X_place)
        else:
            X_place = tf.placeholder(tf.float32, shape=[1, 3, 32, 32])
            pred = model(X_place)
        for i in range(m):
            logits_op = sess.run(pred,feed_dict={X_place:X_adv[i:i+1,:,:,:]})
            if(not np.argmax(logits_op) == np.argmax(Y[i,:])):
                cropped_Y.append(Y[i,:])
                cropped_X_adv.append(X_adv[i,:,:,:])
                cropped_X.append(X[i,:,:,:])
        X_adv = np.array(cropped_X_adv)
        X = np.array(cropped_X)
        Y = np.array(cropped_Y)

        f = open(os.path.join(log_path,'Random_Test_%s_%s.p' % (dataset, attack)),'w')

        pickle.dump({"adv_input":X,"adv_labels":Y},f)
        f.close()

    #np.save(os.path.join(PATH_DATA, 'Adv_%s_%s.npy' % (dataset, attack)), X_adv)
    f = open(os.path.join(log_path,'Adv_%s_%s.p' % (dataset, attack)),'w')

    pickle.dump({"adv_input":X_adv,"adv_labels":Y},f)
    f.close()
    _, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the test set: %0.2f%%" % (100.0 * acc))
    l2_diff = np.linalg.norm(
        X_adv.reshape((len(X), -1)) -
        X.reshape((len(X), -1)),
        axis=1
    ).mean()
    print("Average L-2 perturbation size of the %s attack: %0.2f" %
          (attack, l2_diff))
    if(("adapt" in attack) or ("cw-fp" in attack)):
        return (X, X_adv,Y)
    else:
        print(Y.shape)
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
