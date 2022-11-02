
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

class GANDataset(Dataset):
    def __init__(self, root_dir, train = True, transform = T.Compose([]), target_transform = T.Compose([T.ToTensor()]), size=(32,32), sample_per_class=5000):
        split = 'train' if train else 'test'
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
                
        folders = sorted(os.listdir(os.path.join(root_dir,split)))
        with open(os.path.join(root_dir,'labels.json'))as json_file:
            dataset_classes = json.load(json_file)

        self.classes = [] # list of class names
        self.class_to_idx = {}
        temp_data = []
        self.data = [] # images
        self.targets = [] # labels
        
        
        for index, folder in tqdm(enumerate(folders)):
            class_name = dataset_classes[int(folder)]
            self.classes.append(class_name)
            self.class_to_idx[folder] = index
            folder_dir = os.path.join(root_dir, split, folder)
            files = os.listdir(folder_dir)
            if len(files) > sample_per_class:
                files = files[:sample_per_class]
            [temp_data.append(np.array(Image.open(os.path.join(folder_dir, x)).resize(size))) for x in files]
            self.targets.extend([index] * len(files))
        
        self.data = np.array(temp_data)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        
        img = self.data[i]
        
        label = self.target_transform(self.targets[i])
        img = self.transform(img)
        return img, label