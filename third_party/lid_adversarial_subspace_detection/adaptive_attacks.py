from __future__ import absolute_import
from __future__ import print_function

import copy
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from six.moves import xrange
import sys
sys.path.append('../../.')
from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_argmax
from cleverhans.evaluation import batch_eval
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K
import os
import pickle

def adaptive_fgsm(x, predictions, eps, clip_min=None, clip_max=None,
                  log_dir=None, y=None, model_logits = None):
    """
    Computes symbolic TF tensor for the adversarial samples. This must
    be evaluated with a session.run call.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :return: a tensor for the adversarial example
    """

    # Compute loss]
    logits, = predictions.op.inputs

    fingerprint_dir = log_dir
    fixed_dxs = pickle.load(open(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), "rb"))
    fixed_dys = pickle.load(open(os.path.join(fingerprint_dir, "fp_outputs.pkl"), "rb"))

    if y is None:
        # In this case, use model predictions as ground truth
        y = tf.to_float(
            tf.equal(predictions,
                     tf.reduce_max(predictions, 1, keep_dims=True)))

    output = logits
    pred_class = tf.argmax(y,axis=1)
    loss_fp = 0
    [a,b,c] = np.shape(fixed_dys)
    num_dx = b
    target_dys = tf.convert_to_tensor(fixed_dys)
    target_dys = (tf.gather(target_dys,pred_class))
    norm_logits = output/tf.norm(output)

    for i in range(num_dx):
        logits_p = model_logits(x + fixed_dxs[i])
        logits_p_norm = logits_p/tf.norm(logits_p)
        loss_fp = loss_fp + tf.losses.mean_squared_error((logits_p_norm - norm_logits),target_dys[:,i,:])
        #self appropriate fingerprint


    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss_ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )
    alpha = 0.1
    ## Tune this alpha!!

    loss = loss_ce - alpha*loss_fp

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def adaptive_fast_gradient_sign_method(sess, model, X, Y, eps, clip_min=None,
                              clip_max=None, batch_size=256, log_dir = None,
                                       model_logits = None):
    """
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param eps:
    :param clip_min:
    :param clip_max:
    :param batch_size:
    :return:
    """
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    adv_x = adaptive_fgsm(
        x, model(x), eps=eps,
        clip_min=clip_min,
        clip_max=clip_max, y=y,
        log_dir= log_dir,
        model_logits = model_logits
    )
    X_adv, = batch_eval(
        sess, [x, y], [adv_x],
        [X, Y], feed={K.learning_phase(): 0},
        args={'batch_size': batch_size}
    )
    return X_adv


def adaptive_basic_iterative_method(sess, model, X, Y, eps, eps_iter, nb_iter=50,
                           clip_min=None, clip_max=None, batch_size=256):
    """
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param eps:
    :param eps_iter:
    :param nb_iter:
    :param clip_min:
    :param clip_max:
    :param batch_size:
    :return:
    """
    print("nb_iter",nb_iter)
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,)+X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,)+Y.shape[1:])
    # results will hold the adversarial inputs at each iteration of BIM;
    # thus it will have shape (nb_iter, n_samples, n_rows, n_cols, n_channels)
    results = np.zeros((nb_iter, X.shape[0],) + X.shape[1:])
    # Initialize adversarial samples as the original samples, set upper and
    # lower bounds
    X_adv = X
    X_min = X_adv - eps
    X_max = X_adv + eps
    print('Running BIM iterations...')
    # "its" is a dictionary that keeps track of the iteration at which each
    # sample becomes misclassified. The default value will be (nb_iter-1), the
    # very last iteration.
    def f(val):
        return lambda: val
    its = defaultdict(f(nb_iter-1))
    # Out keeps track of which samples have already been misclassified
    out = set()
    for i in tqdm(range(nb_iter)):
        adv_x = adaptive_fgsm(
            x, model(x), eps=eps_iter,
            clip_min=clip_min, clip_max=clip_max, y=y
        )
        X_adv, = batch_eval(
            sess, [x, y], [adv_x],
            [X_adv, Y], feed={K.learning_phase(): 0},
            args={'batch_size': batch_size}
        )
        X_adv = np.maximum(np.minimum(X_adv, X_max), X_min)
        results[i] = X_adv
        # check misclassifieds
        predictions = model.predict_classes(X_adv, batch_size=512, verbose=0)
        misclassifieds = np.where(predictions != Y.argmax(axis=1))[0]
        for elt in misclassifieds:
            if elt not in out:
                its[elt] = i
                out.add(elt)

    return its, results
