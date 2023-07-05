import numpy as np
from torch.utils.data import Dataset
import torch

class VIS_DATASET(Dataset):
    def __init__(self, a):
        print(a)
        a = np.array(a)
        coor = []
        for k in range(1,3):
            init_coor = np.where(a==k)
            for i in range(len(init_coor[0])):
                coor.append([[init_coor[0][i]/(len(a)), init_coor[1][i]/(len(a))],k-1])

        self.coor = coor
    
    def __len__(self):
        return len(self.coor)
        
    def __getitem__(self,idx):
        data = torch.tensor(self.coor[idx][0]).float()
        label = torch.tensor([self.coor[idx][1]]).float()
        
        return data, label