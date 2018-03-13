"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import pickle

class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func, log_dir):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    if loss_func == 'xent':
      loss = model.xent

    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.logits, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.logits, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)

    elif (loss_func == 'cw_custom' or loss_func=='xent_custom'):
      #fix this later
      if(1>0):
	      label_mask = tf.one_hot(model.y_input,
		                      10,
		                      on_value=1.0,
		                      off_value=0.0,
		                      dtype=tf.float32)
	      correct_logit = tf.reduce_sum(label_mask * model.logits, axis=1)
	      wrong_logit = tf.reduce_max((1-label_mask) * model.logits, axis=1)
	      wrong_logit_arg = tf.argmax(((1-label_mask) * model.logits), axis=1)

	      """
	       Unnecessary, but tensorflow seems to break if we call
	       fixed_dys[wrong_logit_arg,item,:] later
	      """
	      wrong_logit_label_mask = tf.one_hot(wrong_logit_arg,
		                      10,
		                      on_value=1.0,
		                      off_value=0.0,
		                      dtype=tf.float32)

	      wrong_index = tf.tensordot(tf.cast(wrong_logit_label_mask,dtype=tf.float64),
		            1.0*np.array(range(10)),1)
      if(loss_func == 'cw_custom'):
              loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
      if(loss_func == 'xent_custom'):
              loss = model.xent

      # Add loss that makes it align fingerprints also to the wrong class
      fixed_dxs = pickle.load(open(os.path.join(log_dir, "fp_inputs_dx.pkl"), "rb"))
      fixed_dys = pickle.load(open(os.path.join(log_dir, "fp_outputs.pkl"), "rb"))
      for item, perturbs in enumerate(fixed_dxs):
          perturbs = tf.convert_to_tensor(perturbs,dtype=tf.float32)
          perturbs = tf.reshape(model.x_input, [-1, 1, 28, 28]) + perturbs
          perturbs = tf.reshape(perturbs, [-1, 1, 28, 28])
          #perturbs = tf.reshape(perturbs,[1,-1])
          logits_p = model.model(perturbs)
          logits = model.logits
          dy = logits_p/tf.norm(logits_p) - logits/tf.norm(logits)
          dy_ref = np.zeros((1,10))
          dy_ref = tf.matmul(wrong_logit_label_mask,
                        tf.cast(fixed_dys[:,item,:],dtype=tf.float32))
          loss_perturb = 1.0 * (tf.losses.mean_squared_error(dy,
                                        dy_ref))
          loss = loss - loss_perturb
    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
