import torch
import torch.nn as nn
import sys
from visualize import *
from generate_array import get_grid
from vis_dataset import VIS_DATASET
from model import FCN_only2

init_data_array = get_grid()

train_dataset = VIS_DATASET(init_data_array)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

model = FCN_only2(6)


learning_rate = 1e-3
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 1000
model.train()


def train():
    out_list = []
    lat_list = []
    lab_list = []
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        output, latent = model(inputs)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # print(loss)
        total_loss += loss
        
        optimizer.step()
        out_list.append(output.detach().numpy())
        lat_list.append(latent.detach().numpy())
        lab_list.append(labels.detach().numpy())
    print(t, total_loss)
    return out_list, lat_list, lab_list, loss

qApp = QApplication(sys.argv)

for t in range(epoch+1):
    out,lat,lab, loss = train()
    if t ==0 :
        aw = AnimationWidget(out, lat, lab, t, loss)
    else:
        aw.update_var(out,lat,lab, t, loss)
    
aw.show()
sys.exit(qApp.exec_())
