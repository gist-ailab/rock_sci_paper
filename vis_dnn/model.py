import torch.nn as nn
import math

# class FCN_only2(nn.Module):
#     def __init__(self, num_layer) -> None:
#         super().__init__()
#         self.num_layer = num_layer
#         self.feature = []
#         self.layer_list = nn.Sequential( *([nn.Linear(2,2)] * num_layer))
#         # self.act = nn.LeakyReLU(-1.0)
#         self.last_layer = nn.Linear(2,1)
#         self.act = nn.ReLU()
        
#     def forward(self, x):
#         for i in range(self.num_layer):
#             x = self.layer_list[i](x)
#             self.feature.append(x)
#             x = self.act(x)
#         x = self.last_layer(x)
#         return x, self.feature
 

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
        latent = nn.LeakyReLU(-1.0)(latent)
        output = self.last_layer(latent) 
        return output, latent