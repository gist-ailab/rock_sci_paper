import torch
import torch.nn as nn
import torch.optim as optim

### input
input= torch.tensor([[0.3,0.4]])
label= torch.tensor([[1.0]])

### layer
layer1 = nn.Linear(2,2, bias=False)
layer1.weight=torch.nn.Parameter(torch.tensor([[-1.0,2.0],[-2.0,1.0]]))

layer2 = nn.Linear(2,2, bias=False)
layer2.weight=torch.nn.Parameter(torch.tensor([[0.0,-3.0],[1.0,2.0]]))

layer3 = nn.Linear(2,1, bias=False)
layer3.weight=torch.nn.Parameter(torch.tensor([[4.0,-3.0]]))

### model
# model = nn.Sequential(layer1)
# model = nn.Sequential(layer1, layer2)
model = nn.Sequential(layer1, layer2, layer3)
print(model[0])
exit()
print(model.get_parameter('0.weight'))
# print(model.get_parameter('1.weight'))
# print(model.get_parameter('2.weight'))

### forward
output = model(input)
print(output)

### loss
criterion = nn.MSELoss(reduction="none")

### optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

### backward
optimizer.zero_grad()
loss = criterion(output, label)
print(loss)
# exit()
loss.backward()
optimizer.step()
print(model.get_parameter('0.weight'))
print(model.get_parameter('1.weight'))
print(model.get_parameter('2.weight'))
new_output = model(input)
print(new_output)