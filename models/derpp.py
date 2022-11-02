# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        #FIXME
        # save a list with the already logged tasks (in order to log just a batch of images for each task)
        self.logged_images_task = []

    def observe(self, inputs, labels, not_aug_inputs, current_task_labels, task_number=-1, args=None, tb_logger=None, epoch=-1):

        #log_images
        if tb_logger is not None and task_number not in self.logged_images_task:
            tb_logger.log_images(inputs, args, epoch, task_number, tag='images')
            self.logged_images_task.append(task_number)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        #cross-entropy on the current batch
        loss = self.loss(outputs, labels.long())
        #log_loss
        if tb_logger is not None:
            tb_logger.log_loss(loss.item(), args, epoch, task_number, -1, 'loss/task_cross_entropy')

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buff_mse_loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            #log_loss
            if tb_logger is not None:
                tb_logger.log_loss(buff_mse_loss.item(), args, epoch, task_number, -1, 'loss/buffer_mse')
            loss += buff_mse_loss

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buff_ce_loss = self.args.beta * self.loss(buf_outputs, buf_labels)
            #log loss
            if tb_logger is not None:
                tb_logger.log_loss(buff_ce_loss.item(), args, epoch, task_number, -1 , 'loss/buffer_cross_entropy')
            loss += buff_ce_loss

        loss.backward()
        self.opt.step()
        
        # filters the items in the batch based on the current task
        if current_task_labels != []:
            mask_list = torch.stack([labels == l for l in current_task_labels])
            mask = torch.any(mask_list, dim = 0)
            not_aug_inputs = not_aug_inputs[mask]
            labels = labels[mask]
            outputs = outputs[mask]
        
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item(), outputs.detach()
