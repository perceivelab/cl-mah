# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.perm_mnist import PermutedMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.rot_mnist import RotatedMNIST
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.mnist_360 import MNIST360
from datasets.seq_future_cifar import FutureCIFAR10
from datasets.seq_future_mnist import FutureMNIST
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    PermutedMNIST.NAME: PermutedMNIST,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    RotatedMNIST.NAME: RotatedMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MNIST360.NAME: MNIST360,
    FutureCIFAR10.NAME: FutureCIFAR10,
    FutureMNIST.NAME: FutureMNIST
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)


def get_forward_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a GCL dataset.
    This dataset will be used as auxiliary dataset when primary dataset is instance of FutureCIFAR10
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.forward_dataset in NAMES.keys()
    return NAMES[args.forward_dataset](args)