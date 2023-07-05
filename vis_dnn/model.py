import torch.nn as nn
import math

class FCN_only2(nn.Module):
    def __init__(self, num_layer) -> None:
        super().__init__()
        self.num_layer = num_layer
        
        self.layer_list = nn.Sequential()
        for i in range(1,num_layer+1):
            self.layer_list.add_module("layer_"+str(i), nn.Linear(int(math.pow(2,i)),int(math.pow(2,i+1))))
            self.layer_list.add_module("ReLU_"+str(i), nn.ReLU())
        self.visual_layer = nn.Linear(int(math.pow(2,num_layer+1)), 2)
        self.last_layer = nn.Linear(2,1)
        
        
    def forward(self, x):
        x = self.layer_list(x)
        latent = self.visual_layer(x)
        output = self.last_layer(latent)
        return output, latent
    
    