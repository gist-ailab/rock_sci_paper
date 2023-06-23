import os
import torch
from torch.utils.data import Dataset
import glob
import cv2
from PIL import Image
import numpy as np

class RockScissorsPaper(Dataset):
    def __init__(self, transform=None, path='./DATA'):
        self.classes = 3
        self.transform = transform
        self.img_path = glob.glob(os.path.join(path,'paper/*'))+glob.glob(os.path.join(path,'rock/*'))+glob.glob(os.path.join(path,'scissor/*'))
        self.label_dict = {'rock':0, 'scissor':1, 'paper':2}

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = self.transform(img)
        # print(self.img_path[idx].split('\\'))
        label = self.img_path[idx].split('\\')[-2]
        label = self.label_dict[label]
        return img, label