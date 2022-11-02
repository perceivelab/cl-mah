# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils import create_if_not_exists
from datetime import datetime

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')

    """
    Parameters for Auxiliar Datasets
    """
    # used for future_seq_future
    parser.add_argument('--dataset_2', type=str, default='CIFAR100',
                        help='Auxiliar dataset.')
    parser.add_argument('--forward_dataset', type= str, default ='seq-cifar10',
                        help='The type of dataset used to define the heads for the next task.')
    # load a different dataset for each task (aux datasets: 1/, 2/, 3/, etc..)
    parser.add_argument('--multiple_aux_datasets', action='store_true',
                        help='Whether to use a different dataset for each task or not.')
    # save logits
    parser.add_argument('--save_logits', action='store_true',
                        help="Whether storing the logits during training or not")
    parser.add_argument('--logits_path', type=str, default = os.path.join('E:','continualfuture'))
    # next-head selection (only for seq-future-cifar10)
    parser.add_argument('--next_heads', choices = ['random', 'most_activated'], default = 'most_activated' )
    # number of additional auxiliary class used during training
    parser.add_argument('--add_aux_classes', type=int, default=0,
                        help="Number of additional auxiliary classes used during training.")


    # whether using a backbone pretrained or not
    parser.add_argument('--backbone_pretrained', type=str, default=None, help='Where to search for weights for a pretrained backbone')

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            if args.dataset in ['future-seq-cifar10']:
                best = best_args['seq-cifar10'][args.model]
            elif args.dataset in ['future-seq-mnist']:
                best = best_args['seq-mnist'][args.model]
            else:
                best = best_args[args.dataset][args.model]
            
                
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)
    
    if args.multiple_aux_datasets:
        args.dataset_2 = '1'
    
    now = datetime.now()
    args.experiment_name = args.experiment_name + str(now.strftime("%Y%m%d_%H%M%S_%f"))


    if args.savecheck:
        create_if_not_exists('checkpoints/'+ args.experiment_name)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)

    #get the backbone model adding additional auxiliary classes(optional)
    backbone = dataset.get_backbone(args.add_aux_classes)
    
    #using a pretrained backbone
    if args.backbone_pretrained is not None:
        print('Loading backbone weights... ')
        print(backbone.load_state_dict(torch.load(args.backbone_pretrained)))
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    
    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
