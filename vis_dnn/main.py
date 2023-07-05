import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display


from generate_array import get_grid
from vis_dataset import VIS_DATASET
from model import FCN_only2

init_data_array = get_grid()
# print(init_data_array)
# exit()
train_dataset = VIS_DATASET(init_data_array)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

model = FCN_only2(6)
print(model)



learning_rate = 1e-4
loss_function = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 10000
model.train()

            
for t in range(epoch+1):
    # f, axes = plt.subplots(1,2)
    # axes[0].set_facecolor('black')
    # axes[1].set_facecolor('black')
    # f.set_size_inches((50,50))
        
    for i, (inputs, labels) in enumerate(train_loader):
        output, latent = model(inputs)
        latent = latent.detach().numpy()
        
        # if labels.item() == 0:
        #     axes[0].scatter(x = latent[0][0], y = latent[0][1], color='red', s= 100)
        #     axes[1].scatter(output.detach().numpy(), y = 0, color = 'red', s=100)
        # elif labels.item() == 1:
        #     axes[0].scatter(x =latent[0][0], y = latent[0][1], color='yellow', s=100)
        #     axes[1].scatter(output.detach().numpy(), y = 0, color = 'yellow', s=100)
        
    
        # exit()
        
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()
    # f.savefig('s'+str(t)+'.png')
    # f.clf()
    # exit()
    # plt.show()
