import torch
import torch.nn as nn
import sys
from visualize import *
from generate_array import get_grid
from vis_dataset import VIS_DATASET
from model import FCN_only2
from get_train import xor_train

### get mouse input with pygame 

init_data_array = get_grid()

### preprocess dataset and define loader

train_dataset = VIS_DATASET(init_data_array)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

### define model and hyperparameter can change model layer 

model = FCN_only2(7)

learning_rate = 1e-2
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.5)
epoch = 100

### visualize with PyQt5 start

out = [0 for _ in range(len(train_dataset))]
lat = [[[train_dataset[i][0].detach().numpy()]] for i in range(len(train_dataset))]
lab = [train_dataset[i][1].detach().numpy() for i in range(len(train_dataset))]
t = 0
loss = 0

qApp = QApplication(sys.argv)
aw = VisualizeAllLayer(out, lat, lab, t, loss)

### train and visualize

model.train()

for t in range(epoch+1):
    output, feature, label, loss = xor_train(train_loader, model, loss_function, optimizer, scheduler, t)
    aw.update_var(output, feature, label, t, loss)
    
# aw.show()
sys.exit(qApp.exec_())
