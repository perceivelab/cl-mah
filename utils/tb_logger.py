# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.conf import base_path
import os
from argparse import Namespace
from typing import Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str,
                 stash: Dict[Any, str]=None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')
        self.loggers = {}
        self.name = stash['model_name']
        self.experiment_name = args.experiment_name
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(base_path(), 'new_tensorboard_runs', self.experiment_name, a_setting,self.name),
                purge_step=stash['task_idx'] * args.n_epochs + stash['epoch_idx']+1)
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_')])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)

    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common, mean_acc_task_il = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il\
                if setting == 'task-il' else mean_acc_common
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.n_epochs)
            a_logger.add_scalar('acc_mean', mean_acc, task_number * args.n_epochs)
    
    def log_class_distribution(self, labels: torch.Tensor, args: Namespace, n_tasks: int, n_classes_per_task: int,
                 task_number: int, epoch: int) -> None:
        """
        Logs the distribution of the labels in the buffer.

        Parameters
        ----------
        labels : torch.Tensor
        """
            
        class_items = [torch.count_nonzero(labels==i).item() for i in range(n_tasks * n_classes_per_task + args.add_aux_classes)]
        bins = torch.arange(n_tasks * n_classes_per_task + args.add_aux_classes)
        fig = plt.figure(figsize=(10,6))
        plt.bar(bins, class_items)
        
        for a_logger in self.loggers.values():
            #a_logger.add_histogram('buffer_label', labels, task_number * args.n_epochs)
            a_logger.add_figure('buffer class distribution', fig, global_step = task_number * args.n_epochs+ epoch)

    def log_images(self, batch: torch.Tensor, args: Namespace, epoch: int, task_number: int, tag:str):

        grid = torchvision.utils.make_grid(batch, normalize=True)
        for a_logger in self.loggers.values():       
            a_logger.add_image(tag, grid, global_step = task_number * args.n_epochs + epoch)

    def log_image(self, image, args: Namespace, epoch: int, task_number: int, tag:str):
        for a_logger in self.loggers.values():       
            a_logger.add_figure(tag, image, global_step = task_number * args.n_epochs + epoch)
    
    def log_fake_images(self, batch: torch.Tensor, args: Namespace, task_number: int, epoch:int):
        
        """
        Logs fake images. 
        
        """
        for a_logger in self.loggers.values():
            a_logger.add_images('fake_images', batch, global_step = task_number * args.n_epochs+ epoch)
            

    def log_loss(self, loss: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int, tag='loss') -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar(tag, loss, task_number * args.n_epochs + epoch)

    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)

    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()
