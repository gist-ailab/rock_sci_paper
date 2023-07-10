import torch
import torch.nn as nn
import sys
from visualize import *
from generate_array import get_grid
from vis_dataset import *
from model import FCN_exp2, WeightRotater
from get_train import *
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

### get mouse input with pygame 
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed_all(seed)


init_data_array = get_grid()
# init_data_array = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
#                    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0], 
#                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0], 
#                    [0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
#                    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], 
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
print("GRID:\n", init_data_array)

### preprocess dataset and define loader

train_dataset = VIS_DATASET(init_data_array)
all_dot = DOT_DATASET(int(len(init_data_array)/2))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
all_loader = torch.utils.data.DataLoader(all_dot, batch_size = 1, shuffle= False)

### define model and hyperparameter can change model layer 

num_layer = 8
# act_function = "ReLU"
act_function = "LeakyReLU"
# constrains = WeightRotater()
model = FCN_exp2(num_layer, act_function)
# model.apply(constrains)

learning_rate = 1e-4
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=1)

epoch = 1000

### visualize with PyQt5 start
lat = [[[train_dataset[i][0].detach().numpy()]] for i in range(len(train_dataset))]
lab = [train_dataset[i][1].detach().numpy() for i in range(len(train_dataset))]

qApp = QApplication(sys.argv)
aw = LightVisualize(lat, lab, 1)

### train and visualize
for t in range(epoch+1):
    loss = train(train_loader, model, loss_function, optimizer, scheduler)
    threshold = get_threshold(train_loader, model)
    dot_output, dot_feature, train_out, train_feature, train_label = test(all_loader, train_loader, model)
    aw.update_var(train_feature, dot_feature, train_label, t, loss, threshold)
        
sys.exit(qApp.exec_())
