# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:16:27 2021

@author: belli
"""

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SVHN(Dataset):
    def __init__(self, root_dir, split = 'train', transform = T.Compose([]), target_transform = T.Compose([T.ToTensor()]), download=True, samples_per_class=5000):
        
        assert (split in ['train', 'test', 'extra'])
        dataset = torchvision.datasets.SVHN(root=root_dir, split=split, transform=None, target_transform=None, download=download)
        labels = list(set(dataset.labels))
        labels.sort()
        
        
        data = []
        targets = []
        index_mask = [[l == index for l in dataset.labels] for index in labels]
        
        for ind in index_mask:
            data.extend(dataset.data[ind][:samples_per_class])
            targets.extend(dataset.labels[ind][:samples_per_class])
        
        #data
        self.data = np.array(data) #shape: [N,C,H,W]
        self.data = np.moveaxis(self.data, 1, -1) #shape: [N,H,W,C]
        #labels
        self.targets = np.array(targets)
        #classes
        self.classes = [str(i) for i in labels]
        #class_to_idx
        self.class_to_idx = {x: i for i, x in enumerate(labels)}
        #transforms
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target