# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset, get_forward_dataset
import sys
import os
import json
import pickle
from copy import deepcopy
from datetime import datetime

from datasets.seq_future_cifar import FutureCIFAR10
from datasets.seq_future_mnist import FutureMNIST
from sklearn.metrics import confusion_matrix
from utils.conf_matrix import plot_confusion_matrix
import numpy as np
from utils.tsne import tsne_plot

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    
    if dataset.NAME == 'future-seq-cifar10' or dataset.NAME == 'future-seq-mnist':
        current_task_labels = dataset.get_task_labels(k)
        current_task_labels.sort()
        index_0, index_1 = current_task_labels
        outputs[:, :index_0] = -float('inf')
        outputs[:, index_0+1:index_1] = -float('inf')
        outputs[:, index_1+1:] = -float('inf')
    
    else:
        outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                   dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, save_logits:bool=False, path=None, args=None, epoch=-1, task_number=-1, tb_logger=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    if save_logits and path is not None:
        output_list = []
        path = os.path.join(path, 'eval')
        if not os.path.isdir(path):
            os.mkdir(path)
    
    
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    all_labels, all_preds, all_outputs = [], [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels, _ = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            
            # accumulate predictions and labels for confusion matrix
            if len(all_labels)==0:
                all_labels.append(labels.detach().cpu().numpy())
                all_preds.append(pred.detach().cpu().numpy())
                all_outputs.append(outputs.detach().cpu().numpy())
            else: 
                all_preds[0] = np.append(all_preds[0], pred.detach().cpu().numpy(), axis=0)
                all_labels[0] = np.append(all_labels[0], labels.detach().cpu().numpy(), axis=0)
                all_outputs[0] = np.append(all_outputs[0], outputs.detach().cpu().numpy(), axis=0)

            if save_logits:
                output_list.append(outputs.clone().detach())

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0) 
        accs_mask_classes.append(correct_mask_classes / total * 100)
    
    # compute confusion matrix
    all_labels = all_labels[0]
    all_preds = all_preds[0]
    class_conf_matrix = confusion_matrix(all_labels, all_preds)
    cm_figure = plot_confusion_matrix(class_conf_matrix, class_names=test_loader.dataset.classes)
    # t-sne
    all_outputs = all_outputs[0]
    tsne_figure = tsne_plot(all_labels, all_outputs, test_loader.dataset.classes)
    if tb_logger is not None:
        tb_logger.log_image(cm_figure, args, epoch, task_number, tag = 'confusion matrix/class-il')
        tb_logger.log_image(tsne_figure, args, epoch, task_number, tag = 't-sne')

    if save_logits:
        outputs = torch.cat(output_list, dim=0)
        torch.save(outputs, os.path.join(path, 'eval_'+str(len(os.listdir(path))+1)+'.pt'))

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    model_stash = create_stash(model, args, dataset)
    
    train_name = model_stash['model_name'].split('/')[-1]
    
    task_classes={}
    
    logits_path = None
    if args.save_logits:
        if not os.path.join(args.logits_path): 
            os.mkdir(args.logits_path)
        logits_path = os.path.join(args.logits_path, train_name)
        if not os.path.isdir(logits_path):
            os.mkdir(logits_path)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
    else:
        tb_logger = None

    dataset_copy = get_dataset(args)
    
    for t in range(dataset.N_TASKS):
        model.net.train()
        if isinstance(dataset_copy, FutureMNIST) or isinstance(dataset_copy, FutureCIFAR10):
            dataset_copy.set_pos_new_tasks(tuple(range(t* dataset_copy.N_CLASSES_PER_TASK, (t+1)*dataset_copy.N_CLASSES_PER_TASK)))
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy, save_logits=args.save_logits, path=logits_path)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        if args.multiple_aux_datasets:
            if t > 0:
                #change aux dataset at each task
                args.dataset_2 = str(t+1)
                dataset_old = deepcopy(dataset)
                dataset = get_dataset(args)
                dataset.pos_to_label = dataset_old.pos_to_label
                dataset.label_to_pos = dataset_old.label_to_pos
                dataset.i = dataset_old.i
                dataset.test_loaders = dataset_old.test_loaders
        '''
        If we are using FutureCIFAR10 as 'dataset', we need to choose which heads assign to the new task.
        '''
        if isinstance(dataset_copy, FutureMNIST) or isinstance(dataset_copy, FutureCIFAR10):
            if t == 0:
                #create forward dataset
                forward_dataset = get_forward_dataset(args)
                _, _ = forward_dataset.get_data_loaders()
                dataset.set_pos_new_tasks(tuple(range(0, dataset.N_CLASSES_PER_TASK)))
            else:
                #how choose the heads for the next task?
                if args.next_heads == 'most_activated':
                    #given the samples of the new task, the most activated heads become the next heads 
                    next_pos_classes = get_next_pos_classes(model, forward_dataset, dataset.get_free_pos(), dataset.additional_classes)
                else: #args.next_heads =='random'
                    #next heads randomly selected
                    next_pos_classes = np.random.choice(dataset.get_free_pos(), 2, False)
                    print('next heads (randomly):', next_pos_classes)
                dataset.set_pos_new_tasks(next_pos_classes)
                
        train_loader, _ = dataset.get_data_loaders()
        
        task_classes[t] = train_loader.dataset.classes
        
        if isinstance(dataset_copy, FutureMNIST) or isinstance(dataset_copy, FutureCIFAR10):
            current_task_labels = dataset.get_current_labels()
            print(train_loader.dataset.class_to_idx)
        else: current_task_labels = []
        
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        
        print(f"Task n: {t} involves {len(train_loader.dataset)} samples.")
        for epoch in range(args.n_epochs):
            output_list = []
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                                  
                    loss, outputs= model.observe(inputs, labels, not_aug_inputs, current_task_labels, task_number=t, tb_logger=tb_logger, args=args, epoch=epoch)
                
                output_list.append(outputs)
                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
            
            if args.save_logits and(epoch % 20 == 0 or epoch == 49):
                outputs = torch.cat(output_list, dim = 0)
                torch.save(outputs, os.path.join(logits_path, 'train_'+str(t)+'_'+str(epoch)+'.pt'))
            
            if args.tensorboard and epoch % 5 == 0:
                if hasattr(model, 'buffer') and hasattr(model.buffer, 'labels'):
                    tb_logger.log_class_distribution(model.buffer.labels, args, dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, t, epoch)
            
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset, args = args, epoch=epoch, task_number=t, tb_logger=tb_logger)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        
        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        if args.savecheck:
            now = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            torch.save(model.state_dict(), 'checkpoints/%s/%s_%s_%d_%d_%s.pt' % (args.experiment_name, model.NAME, dataset.NAME,
                    model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)))
            if 'buffer_size' in model.args:
                with open( 'checkpoints/%s/%s_%s_bufferoni_%d_%d_%s.pkl' % (args.experiment_name,model.NAME, dataset.NAME, model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)), 'wb') as f:
                    pickle.dump(obj=deepcopy(model.buffer), file=f)
            with open('checkpoints/%s/%s_%s_interpr_%d_%d_%s.pkl' % (args.experiment_name, model.NAME, dataset.NAME, model.args.buffer_size if 'buffer_size' in model.args else 0, t, str(now)), 'wb') as f:
                pickle.dump(obj=args, file=f)
            
    
    #save classes_order during tasks
    if args.save_logits:
        with open(os.path.join(logits_path,'task_classes.json'), 'w') as fw:
            json.dump(task_classes, fw)        
            
    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))



def get_next_pos_classes(model: ContinualModel, forward_dataset: ContinualDataset, free_heads: list, additional_aux_classes: int=0):
    status = model.net.training
    forward_train_loader, _ = forward_dataset.get_data_loaders()
    model.net.eval()
    
    list_labels, list_pred = [], []
    
    #Compute outputs next task
    for data in forward_train_loader:
        inputs, labels, _ = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        
        list_pred.append(pred.cpu())
        list_labels.append(labels.cpu())
    
    pred = torch.cat(list_pred)
    labels = torch.cat(list_labels)
    
    #create predictions matrix
    predictions_matrix = torch.ones((forward_dataset.N_CLASSES_PER_TASK, forward_dataset.N_CLASSES_PER_TASK * forward_dataset.N_TASKS + additional_aux_classes),dtype=torch.int) * -1
    #only columns in free_outputs are available
    predictions_matrix[:,free_heads] = 0
    
    next_labels = list(set(forward_train_loader.dataset.targets))
    print(f'{forward_dataset.NAME} next classes: {next_labels}')
    
    for i, label in enumerate (next_labels):
        l_mask = [labels == label]
        all_pred = pred[l_mask].unique().tolist()
        print(f'activated_heads: {all_pred}')
        # possible heads
        possible_outputs = list(set(all_pred) & set(free_heads))

        for l in possible_outputs:
            predictions_matrix[i,l] = (pred[l_mask] == l).sum(dim=0).item()
    print(predictions_matrix)
    
    #define the labels-heads mapping for the next task
    list_pos = [0] * len(next_labels)
    for row in range(predictions_matrix.shape[0]):
        index = (predictions_matrix == torch.max(predictions_matrix)).nonzero(as_tuple=False)[0]
        r = index[0].item()
        c = index[1].item()
        list_pos[r] = c
        predictions_matrix[r,:] = -1
        predictions_matrix[:, c] = -1
    
    print('next heads:', list_pos)
    
    model.net.train(status)
    return list_pos