"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nn_transfer import transfer, util
import torch.nn.functional as F
import sys
import os.path
import warnings
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from third_party.attacks import craft_one_type
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
from model import CW_Net as Net
sys.path.insert(0, './mnist')
sys.path.append('..')
from third_party.lid_adversarial_subspace_detection.util import (get_data, cross_entropy) # ,get_noisy_samples)
from keras import backend as K
from keras.layers import Activation
import util
from keras import backend as K


# Global constants
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--attack', default="fgsm")
parser.add_argument('--ckpt', default="/tmp/user/logs/mnist/ckpt/state_dict-ep_1.pth")
parser.add_argument('--log-dir', type=str, default="/tmp/user/logs/mnist/adv_examples")
parser.add_argument('--fingerprint-dir', type=str, default="/tmp/logs/neural_fingerprint/mnist/eps_0.1/numdx_5")
args = parser.parse_args()

with open('./mnist/config.json') as config_file:
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
    dataset = 'mnist'

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

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

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


if(1>0):
    dataset = 'mnist'


    _, _, X_test, Y_test = get_data(dataset)
    num_samples = np.shape(X_test)[0]
    num_rand_samples = 20
    random_samples = np.random.randint(0,num_samples, num_rand_samples)
    new_X_test = np.zeros((num_rand_samples, 1, 28, 28))
    for i,sample_no in enumerate(random_samples):
            new_X_test[i,0,:,:] = (X_test[sample_no,:,:,0])
    new_Y_test = Y_test[random_samples,:]

    print("attack:", args.attack)

    f = open(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)),'w')
    print(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)))

    pickle.dump({"adv_input":new_X_test,"adv_labels":new_Y_test},f)
    f.close()

with tf.Session() as sess:

    K.set_session(sess)
    K.set_image_data_format('channels_first')
    if(args.attack == 'spsa' or args.attack == 'all'):
        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model = Model(torch_model=pytorch_network)
        keras_network = model.model
        transfer.pytorch_to_keras(pytorch_network, model.model)
        pytorch_network.eval()
        model = model.model
        model_logits = model
        batch_size = 16
        craft_one_type(sess, model, new_X_test, new_Y_test, dataset, 'spsa',
                           batch_size, log_path=args.log_dir,
                           fp_path= args.fingerprint_dir)

    if(args.attack == 'adapt-pgd' or args.attack == 'all'):
        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model = Model(torch_model=pytorch_network)
        keras_network = model.model
        transfer.pytorch_to_keras(pytorch_network, model.model)
        pytorch_network.eval()
        model = model.model
        model_logits = model
        batch_size = 16
        craft_one_type(sess, model, new_X_test, new_Y_test, dataset, 'adapt-pgd',
                           batch_size, log_path=args.log_dir,
                           fp_path= args.fingerprint_dir)


    if(args.attack == 'cw-l2' or args.attack == 'all'):
        #No softmax for Carlini attack
        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model = Model(torch_model=pytorch_network,softmax=False)
        keras_network = model.model
        transfer.pytorch_to_keras(pytorch_network, model.model)
        pytorch_network.eval()
        model = model.model
        batch_size = 16
        craft_one_type(sess, model, new_X_test, new_Y_test, dataset, 'cw-l2',
                           batch_size, log_path=args.log_dir,
                           fp_path= args.fingerprint_dir  )


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

    if(args.attack in ['adapt-fgsm','adapt-all']):
        # FGSM, BIM-a, JSMA

        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model_logits  = Model(torch_model=pytorch_network)
        model  = Model(torch_model=pytorch_network)
        keras_network = model.model
        pytorch_network.eval()
        transfer.pytorch_to_keras(pytorch_network, model.model)
        transfer.pytorch_to_keras(pytorch_network, model_logits.model)
        #util.test_tf2torch( model.model, pytorch_network,(1, 28, 28), num_rand_inp=10, precision=10**-2)
        # Add tests to ensure model is transferred well
        model = model.model
        model_logits = model.model
        if(args.attack == 'adapt-all'):
            for attack in ['adapt-fgsm',"adapt-bim-b"]:
                (X_cropped, X_adv,Y_adv) = craft_one_type(sess, model, new_X_test, new_Y_test, dataset, attack,
                               args.batch_size, log_path=args.log_dir, fp_path= args.fingerprint_dir,
                                               model_logits = model_logits)
        else:

            (X_cropped, X_adv,Y_adv) = craft_one_type(sess, model, new_X_test, new_Y_test, dataset, args.attack,
                               args.batch_size, log_path=args.log_dir, fp_path= args.fingerprint_dir,
                                           model_logits = model_logits)
        f = open(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)),'w')
        print(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)))
        pickle.dump({"adv_input":X_cropped,"adv_labels":Y_adv},f)
        f.close()

    if(args.attack == 'cw-fp' or args.attack == 'all'):
        #No softmax for Carlini attack
        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model = Model(torch_model=pytorch_network,softmax=False)
        keras_network = model.model
        transfer.pytorch_to_keras(pytorch_network, model.model)
        pytorch_network.eval()
        model = model.model
        batch_size = 16
        (X_cropped, X_adv,Y_adv) = craft_one_type(sess, model, new_X_test, new_Y_test, dataset, 'cw-fp',
                           batch_size, log_path=args.log_dir, fp_path= args.fingerprint_dir)

        f = open(os.path.join(args.log_dir,'Random_Test_%s_%s.p' % (dataset)),'w')
        print(os.path.join(args.log_dir,'Random_Test_%s_%s.p' % (dataset)))
        pickle.dump({"adv_input":X_cropped,"adv_labels":Y_adv},f)
        f.close()
    if(args.attack in ['fgsm','bim-a','bim-b','jsma','all']):
        # FGSM, BIM-a, JSMA
        #
        pytorch_network = Net()
        pytorch_network.load_state_dict(torch.load(args_ckpt))
        pytorch_network.eval()
        model = Model(torch_model=pytorch_network)
        keras_network = model.model
        transfer.pytorch_to_keras(pytorch_network, model.model)
        pytorch_network.eval()
        #util.test_tf2torch( model.model, pytorch_network,(1, 28, 28), num_rand_inp=10, precision=10**-2)
        # Add tests to ensure model is transferred well
        model = model.model
        if(args.attack == 'all'):
            for attack in ['fgsm','bim-a','bim-b','jsma']:

                (X_adv,Y_adv) = craft_one_type(sess, model, new_X_test, new_Y_test, dataset, attack,
                               args.batch_size, log_path=args.log_dir)
        else:


            (X_adv,Y_adv) = craft_one_type(sess, model, new_X_test, new_Y_test, dataset, args.attack,
                               args.batch_size, log_path=args.log_dir)


    sess.close()
"""
if(args.attack in ['adapt-bim-b','adapt-all']):
    num_splits = 5
    net_Y = None
    net_X = None
    net_X_adv = None
    for i in range(num_splits):
        K.clear_session()
        with tf.Session() as sess:

            dataset = 'mnist'
            K.set_session(sess)
            K.set_image_data_format('channels_first')

            _, _, X_test, Y_test = get_data(dataset)
            num_samples = np.shape(X_test)[0]
            num_rand_samples = 16
            random_samples = np.random.randint(0,num_samples, num_rand_samples)
            new_X_test = np.zeros((num_rand_samples, 1, 28, 28))
            for i,sample_no in enumerate(random_samples):
                    new_X_test[i,0,:,:] = (X_test[sample_no,:,:,0])
            new_Y_test = Y_test[random_samples,:]

            print("attack:", args.attack)

            f = open(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)),'w')
            print(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)))
            pickle.dump({"adv_input":new_X_test,"adv_labels":new_Y_test},f)
            f.close()

            # FGSM, BIM-a, JSMA
            #
            pytorch_network = Net()
            pytorch_network.load_state_dict(torch.load(args_ckpt))
            pytorch_network.eval()
            model_logits  = Model(torch_model=pytorch_network)
            model  = Model(torch_model=pytorch_network)
            keras_network = model.model
            pytorch_network.eval()
            transfer.pytorch_to_keras(pytorch_network, model.model)
            transfer.pytorch_to_keras(pytorch_network, model_logits.model)
            #util.test_tf2torch( model.model, pytorch_network,(1, 28, 28), num_rand_inp=10, precision=10**-2)
            # Add tests to ensure model is transferred well
            model = model.model
            model_logits = model.model
            split_start = i*(1.0/num_splits)*num_rand_samples
            split_end = (i+1)*(1.0/num_splits)*num_rand_samples
            if(i<num_splits-1):
                (X_cropped, X_adv,Y_adv) = craft_one_type(sess, model, new_X_test[split_start:split_end,:,:,:],
                                               new_Y_test[split_start:split_end,:],
                                               dataset, 'adapt-bim-b',
                                   args.batch_size, log_path=args.log_dir, fp_path= args.fingerprint_dir,
                                                   model_logits = model_logits)
            else:
                (X_cropped, X_adv,Y_adv) = craft_one_type(sess, model, new_X_test[split_start:,:,:,:],
                                               new_Y_test[split_start:,:],
                                               dataset, 'adapt-bim-b',
                                   args.batch_size, log_path=args.log_dir, fp_path= args.fingerprint_dir,
                                                   model_logits = model_logits)
            if(i==0):
                net_Y = Y_adv
                net_X_adv = X_adv
                net_X = X_cropped
            else:
                net_Y = np.concatenate((net_Y, Y_adv),axis=0)
                net_X_adv = np.concatenate((net_X_adv, X_adv),axis=0)
                net_X = np.concatenate((net_X, X_cropped),axis=0)

        f = open(os.path.join(args.log_dir,'Adv_%s_%s.p' % (dataset, attack)),'w')

        pickle.dump({"adv_input":net_X_adv,"adv_labels":net_Y},f)
        f.close()

        f = open(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)),'w')
        print(os.path.join(args.log_dir,'Random_Test_%s_.p' % (dataset)))
        pickle.dump({"adv_input":X_cropped,"adv_labels":net_Y},f)
        f.close()
    sess.close()
"""
