
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

class ChinaMNIST(Dataset):
    def __init__(self, root_dir, train = True, transform = T.Compose([]), target_transform = T.Compose([T.ToTensor()]), size=(32,32), sample_per_class=5000):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
                
        image_paths = sorted(os.listdir(os.path.join(root_dir,'data')))

        self.classes = [ 'c'+str(i) for i in range(15)] # list of class names
        self.class_to_idx = {self.classes[i]:i for i in range(15)}
        temp_data = []
        self.data = [] # images
        self.targets = [] # labels
        
        
        for _,img_path in tqdm(enumerate(image_paths)):
            idx = int(os.path.splitext(img_path)[0].split('_')[-1]) - 1
            class_name = self.classes[idx]
            temp_data.append(np.array(Image.open(os.path.join(root_dir,'data', img_path)).resize(size)))
            self.targets.append(idx)
        
        self.data = np.array(temp_data)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        
        img = self.data[i]
        
        label = self.target_transform(self.targets[i])
        img = self.transform(img)
        return img, label
