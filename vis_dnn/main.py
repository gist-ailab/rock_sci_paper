import torch
import torch.nn as nn
import sys
from visualize import *
from generate_array import get_grid
from vis_dataset import VIS_DATASET
from model import FCN_exp2
from get_train import exp_train

### get mouse input with pygame 

init_data_array = get_grid()

### preprocess dataset and define loader

train_dataset = VIS_DATASET(init_data_array)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

### define model and hyperparameter can change model layer 

model = FCN_exp2(7)

learning_rate = 1e-3
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.5)

epoch = 1000

### visualize with PyQt5 start
qApp = QApplication(sys.argv)


### train and visualize

model.train()

for t in range(epoch+1):
    output, feature, label, loss = exp_train(train_loader, model, loss_function, optimizer, scheduler)
    
    if t ==0 :
        aw = ExperimentWidget(output, feature, label, t, loss)
    else:
        aw.update_var(output, feature, label, t, loss)
    
sys.exit(qApp.exec_())
