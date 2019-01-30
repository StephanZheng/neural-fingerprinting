""" Code to build all data loader for outlier experiments"""
import sys

import torchvision

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.mister_ed.utils import pytorch_utils as utils
import src.mister_ed.config as config
import os

###############################################################################
#                           PARSE CONFIGS                                     #
###############################################################################

DEFAULT_DATASETS_DIR = config.DEFAULT_DATASETS_DIR
DEFAULT_BATCH_SIZE = config.DEFAULT_BATCH_SIZE
DEFAULT_WORKERS = config.DEFAULT_WORKERS
CIFAR10_MEANS = config.CIFAR10_MEANS
CIFAR10_STDS = config.CIFAR10_STDS


##############################################################################
#                                                                            #
#                               DATA LOADER                                  #
#                                                                            #
##############################################################################

def load_cifar10_data(train_or_val, extra_args=None, dataset_dir=None,
                      normalize=False, batch_size=None, manual_gpu=None,
                      shuffle=True, no_transform=True):
    """ Builds a CIFAR10 data loader for either training or evaluation of
        CIFAR10 data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}
    constructor_kwargs.update(extra_args or {})

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)
    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataset_dir, train=train_or_val == 'val',
                         transform=transform_chain, download=True),
        **constructor_kwargs)


def load_cifar100_data(train_or_val, extra_args=None, dataset_dir=None,
                       normalize=False, batch_size=None, manual_gpu=None,
                       shuffle=True, no_transform=True):
    """ Builds a CIFAR100 data loader for either training or evaluation of
        CIFAR100 data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}
    constructor_kwargs.update(extra_args or {})

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)
    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
        datasets.CIFAR100(root=dataset_dir, train=train_or_val == 'val',
                          transform=transform_chain, download=True),
        **constructor_kwargs)


def load_svhn_data(train_or_val, extra_args=None, dataset_dir=None,
                   normalize=False, batch_size=None, manual_gpu=None,
                   shuffle=True, no_transform=True):
    """ Builds a SVHN data loader for either training or evaluation of
        SVHN data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}
    constructor_kwargs.update(extra_args or {})

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)
    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
        datasets.SVHN(root=dataset_dir, split='test',
                      transform=transform_chain, download=True),
        **constructor_kwargs)


def load_lsun_data(dataset_dir=None, normalize=False, batch_size=None, manual_gpu=None,
                   shuffle=True, no_transform=True):
    """ Builds a LSUN(resized) data loader for either training or evaluation of
        LSUN data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    dataset_dir = os.path.join(dataset_dir, "/LSUN_resize/")
    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################

    testsetout = torchvision.datasets.ImageFolder(dataset_dir, transform=transform_chain)
    testloaderOut = torch.utils.data.DataLoader(testsetout, **constructor_kwargs)

    return testloaderOut


def load_imagenet_data(dataset_dir=None, normalize=False, batch_size=None, manual_gpu=None,
                       shuffle=True, no_transform=True):
    """ Builds a tiny imagenet(resized) data loader for either training or evaluation of
        ImageNet data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    dataset_dir = os.path.join(dataset_dir, "/ImageNet_resize/")
    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################

    testsetout = torchvision.datasets.ImageFolder(dataset_dir, transform=transform_chain)
    testloaderOut = torch.utils.data.DataLoader(testsetout, **constructor_kwargs)

    return testloaderOut


def load_place365_data(dataset_dir=None, normalize=False, batch_size=None, manual_gpu=None,
                       shuffle=True, no_transform=True):
    """ Builds a tiny imagenet(resized) data loader for either training or evaluation of
        ImageNet data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
        no_transform: boolean - if True, we don't do any random cropping/
                                reflections of the data
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    dataset_dir = os.path.join(dataset_dir, "/val_32/")
    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}

    # transform chain
    transform_list = []
    if no_transform is False:
        transform_list.extend([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4)])
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[1.0, 1.0, 1.0])
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################

    testsetout = torchvision.datasets.ImageFolder(dataset_dir, transform=transform_chain)
    testloaderOut = torch.utils.data.DataLoader(testsetout, **constructor_kwargs)

    return testloaderOut
