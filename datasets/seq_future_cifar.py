# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:07:30 2021

@author: belli
"""

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from datasets.seq_cifar10 import SequentialCIFAR10, MyCIFAR10
from datasets.cifar100_coarse import CIFAR100Coarse
from datasets.gan_dataset import GANDataset
from datasets.svhn import SVHN
from datasets.seq_tinyimagenet import base_path
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import store_previous_masked_loaders
from argparse import Namespace


import os
import numpy as np
                
        
class JoinedDataset(MyCIFAR10):
       
    FAKE_CLASS_LABELS = []
    FAKE_CLASS_NAMES = []
    
    def __init__(self, root_1:str, root_2:str, train:bool = True, transform:transforms=None, target_transform=None, download:bool=False, labels=[0,1,None,None,None,None,None,None,None,None]) -> None:
        super(JoinedDataset, self).__init__(root_1, train, transform, target_transform, download)
        
        #Auxiliar Datasets
        d_name = os.path.split(root_2)[-1]
        if d_name == 'CIFAR100':
            dataset_2 = CIFAR100Coarse(root_2, train, transform, target_transform, download)
        elif d_name == 'CIFAR10': 
            dataset_2 = CIFAR10(root_2, train, transform, target_transform, download)
        elif d_name == 'SVHN':
            samples_per_class = 5000
            dataset_2 = SVHN(root_2, 'extra', transform, target_transform, download, samples_per_class)
        else:    
            dataset_2 = GANDataset(root_2, train, transform)
        
            
        num_classes_1 = len(self.classes)
        num_classes_2 = len(dataset_2.classes)
                
        assert num_classes_2 >= num_classes_1, 'dataset2 num_classes too small'
        
        #select len(labels) fake classes from dataset_2
        if len(JoinedDataset.FAKE_CLASS_LABELS) == 0:
            if num_classes_2 > num_classes_1:
                JoinedDataset.FAKE_CLASS_LABELS = np.random.choice(num_classes_2, len(labels), replace=False)
            else:
                JoinedDataset.FAKE_CLASS_LABELS = np.arange(num_classes_2)
            #save fake_classes names
            JoinedDataset.FAKE_CLASS_NAMES = [dataset_2.classes[l] for l in JoinedDataset.FAKE_CLASS_LABELS]
        
        '''
        Prepare data of the original dataset
        '''
        index_mask = [t in labels for t in self.targets]
        data_1 = self.data[index_mask]
        targets_1 = np.array(self.targets)[index_mask]
        '''
        updates targets values:
            1. store original_indexes
            2. change original_indexes with used_indexes according to labels param
        '''
        # 1. store original_indexes
        swap_dict = {}
        for l in labels:
            if l is not None and l != labels.index(l):
                swap_dict[l] = [targets_1 == l]
        # 2. change original_indexes with used_indexes
        for k, v in swap_dict.items():
            targets_1[tuple(v)] = labels.index(k)
        
        # save new classes list
        classes_1 = [self.classes[l] if l is not None else None for l in labels]
        
        '''
        Fill dataset with auxiliar data
        '''
        fake_labels = [JoinedDataset.FAKE_CLASS_LABELS[i] if labels[i] == None else None for i in range(len(labels))]
        index_mask = [t in fake_labels for t in dataset_2.targets]
        data_2 = dataset_2.data[index_mask]
        targets_2 = np.array(dataset_2.targets)[index_mask]
        
        swap_dict_2 = {}
        for l in fake_labels:
            if l is not None:
                swap_dict_2[l] = [targets_2 == l]
        for k, v in swap_dict_2.items():
            targets_2[tuple(v)] = fake_labels.index(k)
        
        self.data = np.concatenate([data_1, data_2])
        self.targets = np.concatenate([targets_1, targets_2])
        self.classes = [dataset_2.classes[x] if x is not None else classes_1[i] for i, x in enumerate(fake_labels)]
        self.class_to_idx = {x: i for i, x in enumerate(self.classes)}
        




class FutureCIFAR10(SequentialCIFAR10):
    NAME = 'future-seq-cifar10'
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        
        '''
        number of additional auxiliary classes used during training
        '''
        self.additional_classes = args.add_aux_classes
        '''
        List of lists. Keeps track of the original class subdivision according to the task. 
        '''
        self.classes_per_task = [[j for j in range(i, i+self.N_CLASSES_PER_TASK)] for i in range(0, self.N_CLASSES_PER_TASK * self.N_TASKS, self.N_CLASSES_PER_TASK)] 
        '''
        This dict stores the mapping between the label used by the label and the original class label.
        key: new label
        value: old label
        '''
        self.pos_to_label = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK + self.additional_classes)}
        '''
        This dict stores the mapping between the original class label and the label used by the model. 
        key: old label
        value: new label
        '''
        self.label_to_pos = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK + self.additional_classes)}
        
        
        
    def set_pos_new_tasks(self, pos: list) -> None:
        assert len(pos) == self.N_CLASSES_PER_TASK
        num_task = self.i // self.N_CLASSES_PER_TASK
        classes = self.classes_per_task[num_task]
        for p, c in zip(pos, classes):
            self.pos_to_label[p] = c
            self.label_to_pos[c] = p
    
    
    def get_free_pos(self):
        free_pos = [k for k, v in self.pos_to_label.items() if v is None]
        return free_pos
    
    
    def get_data_loaders(self):
        
        transform = self.TRANSFORM

        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = JoinedDataset(base_path() + 'CIFAR10', base_path() + self.args.dataset_2, train=True, download=True, transform=transform, labels = list(self.pos_to_label.values()))
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = JoinedDataset(base_path() + 'CIFAR10', base_path() + self.args.dataset_2, train=False, download=True, transform=transform, labels = list(self.pos_to_label.values()))

        train, test = store_previous_masked_loaders(train_dataset, test_dataset, self)
        return train, test
        
    
    def get_current_labels(self):
        
        t = self.i // self.N_CLASSES_PER_TASK
        return list(self.label_to_pos.values())[(t-1)*self.N_CLASSES_PER_TASK : t*self.N_CLASSES_PER_TASK]
        
    def get_task_labels(self, t:int):
        
        return list(self.label_to_pos.values())[t*self.N_CLASSES_PER_TASK : (t+1)*self.N_CLASSES_PER_TASK]
        
    

class FutureFillerCIFAR10(SequentialCIFAR10):
    NAME = 'futurefill-seq-cifar10'
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        
        '''
        List of list. Keeps track of the original class subdivision according to the task. 
        '''
        self.classes_per_task = [[j for j in range(i, i+self.N_CLASSES_PER_TASK)] for i in range(0, self.N_CLASSES_PER_TASK * self.N_TASKS, self.N_CLASSES_PER_TASK)] 
        '''
        This dict stores the mapping between the label used by the label and the original class label.
        key: new label
        value: old label
        '''
        self.pos_to_label = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK)}
        '''
        This dict stores the mapping between the original class label and the label used by the model. 
        key: old label
        value: new label
        '''
        self.label_to_pos = {k: None for k in range(self.N_TASKS * self.N_CLASSES_PER_TASK)}
        
        
    def set_pos_new_tasks(self, pos: list) -> None:
        assert len(pos) == self.N_CLASSES_PER_TASK
        num_task = self.i // self.N_CLASSES_PER_TASK
        classes = self.classes_per_task[num_task]
        for p, c in zip(pos, classes):
            self.pos_to_label[p] = c
            self.label_to_pos[c] = p
    
    
    def get_free_pos(self):
        free_pos = [k for k, v in self.pos_to_label.items() if v is None]
        return free_pos
    
    
    def get_data_loaders(self):
        
        transform = self.TRANSFORM

        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])
        train_dataset = JoinedFillerDataset(base_path() + 'CIFAR10', base_path() + self.args.dataset_2, train=True, download=True, transform=transform, labels = list(self.pos_to_label.values()))
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = JoinedFillerDataset(base_path() + 'CIFAR10', base_path() + self.args.dataset_2, train=False, download=True, transform=transform, labels = list(self.pos_to_label.values()))

        train, test = store_previous_masked_loaders(train_dataset, test_dataset, self)
        return train, test
        
    
    def get_current_labels(self):
        
        t = self.i // self.N_CLASSES_PER_TASK
        return list(self.label_to_pos.values())[(t-1)*self.N_CLASSES_PER_TASK : t*self.N_CLASSES_PER_TASK]
        
    def get_task_labels(self, t:int):
        
        return list(self.label_to_pos.values())[t*self.N_CLASSES_PER_TASK : (t+1)*self.N_CLASSES_PER_TASK]
        
        
        
        
        