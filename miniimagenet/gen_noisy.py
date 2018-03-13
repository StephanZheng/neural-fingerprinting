__author__ = 'user'
# First train a model without fingerprint, generate adversarials
# Generate noise corresponding to all adversarials
# Mix noisy data with train-data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nn_transfer import transfer, util
import json
import argparse
import math
import os
import torch
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf_model import Model
from pgd_cw_whitebox import LinfPGDAttack
import dill as pickle
from model import CW2_Net as Net
from keras import backend as K
from keras.layers import Activation
import sys
sys.path.insert(0, './cifar/')
#Set path for attack code
sys.path.insert(0, './mnist/')
from third_party.lid_adversarial_subspace_detection.util import (get_data, cross_entropy)
from attacks import craft_one_type
# Global constants
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--attack', default="fgsm")
parser.add_argument('--batch-size',type=int, default=128)
parser.add_argument('--ckpt', default="/tmp/user/logs/cifar/ckpt/state_dict-ep_1.pth")
parser.add_argument('--log-dir', type=str, default="/tmp/user/logs/cifar/noisy_train_data/")
args = parser.parse_args()

with open('./cifar/config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
args_ckpt = args.ckpt
model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


#global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

# Transfer Model From Pytorch to TensorFlow




# A function for evaluating a single checkpoint
def evaluate_checkpoint(sess,model):
    dataset = 'cifar'

    #with tf.Session() as sess:
    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    adv_x_samples=[]
    adv_y_samples=[]
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist.test.images[bstart:bend,:]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      if(ibatch == 0):
          adv_x_samples = x_batch_adv
          adv_y_samples = y_batch
      else:
          adv_x_samples = np.concatenate((adv_x_samples, x_batch_adv), axis = 0)
          adv_y_samples = np.concatenate((adv_y_samples, y_batch), axis = 0)
    if(args.attack == 'xent'):
      atck = 'pgd'
      f = open(os.path.join(args.log_dir, 'Adv_%s_%s.p' % (dataset, atck)), "w")
    elif(args.attack == 'cw_pgd'):
      atck = 'cw_pgd'
      f = open(os.path.join(args.log_dir, 'Adv_%s_%s.p' % (dataset, atck)), "w")
    else:
      f = open(os.path.join(args.log_dir, "custom.p"), "w")
    pickle.dump({"adv_input":adv_x_samples,"adv_labels":adv_y_samples},f)
    f.close()


with tf.Session() as sess:
    dataset = 'cifar'
    K.set_session(sess)
    K.set_image_data_format('channels_first')

    # Sample random test data
    _, _, X_test, Y_test = get_data(dataset)
    num_samples = np.shape(X_test)[0]
    num_rand_samples = 1328
    random_samples = np.random.randint(0,num_samples, num_rand_samples)
    new_X_test = X_test[random_samples,:,:,:]
    new_Y_test = Y_test[random_samples,:]

    f = open(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)),'w')
    pickle.dump({"adv_input":new_X_test,"adv_labels":new_X_test},f)
    f.close()

    if(args.attack == 'cw-l2' or args.attack == 'all'):
        pytorch_network = Net()
        model = Model(torch_model=pytorch_network,softmax=False)
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        transfer.pytorch_to_keras( pytorch_network, model.model)
        #No softmax for Carlini attack
        model = model.model
        batch_size = 16
        _, acc = model.evaluate(new_X_test, new_Y_test, batch_size=16,
                                verbose=0)
        craft_one_type(sess, model, new_X_test, new_Y_test, dataset, 'cw-l2',
                           batch_size, log_path=args.log_dir)

    if(args.attack == 'xent' or args.attack == 'cw_pgd'):
        ########## NOTE: Enable custom attacks if needed later, code available

        #model.model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD())
        # PGD based attacks
        if eval_on_cpu:
          with tf.device("/cpu:0"):
            attack = LinfPGDAttack(model,
                                   config['epsilon'],
                                   config['k'],
                                   config['a'],
                                   config['random_start'],
                                   args.attack,
                                   config['log_dir'])
        else:
          attack = LinfPGDAttack(model,
                                 config['epsilon'],
                                 config['k'],
                                 config['a'],
                                 config['random_start'],
                                   args.attack,
                                 config['log_dir'])
        evaluate_checkpoint(sess,model)

    if(args.attack in ['fgsm','bim-a','bim-b','jsma','all']):
        #Transfer model from torch to tf
        pytorch_network = Net()
        model = Model(torch_model=pytorch_network, softmax=True)
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        transfer.pytorch_to_keras( pytorch_network, model.model)
        # Add tests to ensure model is transferred well
        # FGSM, BIM-a, JSMA

        model = model.model
        if args.attack == 'all':
            # Cycle through all attacks
            for attack in ['fgsm','bim-a','bim-b','jsma']:
                craft_one_type(sess, model, new_X_test, new_Y_test, dataset, attack,
                               args.batch_size, log_path=args.log_dir)
        else:
                craft_one_type(sess, model, new_X_test, new_Y_test, dataset, args.attack,
                               args.batch_size, log_path=args.log_dir)

    sess.close()
