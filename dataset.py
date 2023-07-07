import os
import torch
from torch.utils.data import Dataset
import glob
import cv2
from PIL import Image
import numpy as np

class RockScissorsPaper(Dataset):
    def __init__(self, transform=None, path='./DATA', mode='train'):
        self.class_num = 3
        self.mode = mode
        self.transform = transform
        self.rootpath = path
        self.img_path = glob.glob(os.path.join(path,'paper/*'))+glob.glob(os.path.join(path,'rock/*'))+glob.glob(os.path.join(path,'scissors/*'))
        self.label_dict = {'rock':0, 'scissors':1, 'paper':2}
        self.names = []
        with open(os.path.join(self.rootpath, f'{mode}.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.names.append(line)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name_path = os.path.join(self.rootpath, self.names[idx])
        img = cv2.imread(name_path)
        img = self.transform(img)
        label = name_path.split('\\')[-2]
        label = self.label_dict[label]
        return img, label