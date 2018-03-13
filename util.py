from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import keras
from keras import backend as K
import numpy as np
import dill as pickle
import numpy as np

def test_tf2torch(tf_model,torch_model,input_shape, num_rand_inp=10, precision=10**-2):
    """
    Checks consistency of torch and tf models before generating attacks
    :param tf_model: copied tf model
    :param torch_model: torch model to be transferred to tf
    :param input_shape: Format Channels X Height X Width
    :param num_rand_inp: number of random inputs to test consistency on
    :return: raises error if the outputs are not consistent
    """
    torch_model.eval()
    rand_x = torch.rand(num_rand_inp,input_shape[0],input_shape[1],input_shape[2])
    tf_op = tf_model.predict(rand_x.numpy())
    torch_op = F.softmax(torch_model(Variable(rand_x))).data.numpy()
    assert tf_op.shape == torch_op.shape, "Mismatch of dimensions of the outputs from tf and torch models"
    assert np.linalg.norm(torch_op-tf_op)/np.linalg.norm(torch_op)<=num_rand_inp*precision, "Outputs of the torch and tensorflow models" \
                                                            "do not agree"
    pass

def np2var(x,cuda):
    if cuda:
        return Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor))
    else:
        return Variable(torch.from_numpy(x).type(torch.FloatTensor))

def var2np(x,cuda):
    t = x.data
    if cuda:
        t = t.cpu()
    return t.numpy()

def t2var(t,cuda):
    if cuda:
        t = t.cuda()
    t = Variable(t, volatile=True)
    return t

def t2np(t,cuda):
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

def torch2tf(torch_Net,tf_Net):

    return tf_Net