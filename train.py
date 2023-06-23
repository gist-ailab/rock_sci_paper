# 필요한 라이브러리 import
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import dataset

# 각종 path 및 파라미터 설정
data_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\LR_ro_sci_pa'
# save_path = '모델 파라미터를 저장할 디렉토리 경로'
epochs = 30
batch_size = 32
learning_rate = 0.01

# gpu를 사용할 수 있으면 gpu를 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

# dataset 설정
datasets = dataset.RockScissorsPaper(
    transform=transform,
    path = data_path
)
num_data = len(datasets)
num_train = int(num_data*0.7)
num_test = num_data - num_train

train_data, test_data = torch.utils.data.random_split(datasets, [num_train, num_test])

# dataloader 설정
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 모델, 손실함수, 옵티마이저 설정
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 3)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(epoch):
    print('\nEpoch: %d'%epoch)
    # model train mode로 전환
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    total_loss = running_loss / len(trainloader)
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
    
def test(epoch):
    print('\nEpoch: %d'%epoch)
    # model eval mode로 전환
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        total_loss = running_loss / len(testloader)
        total_acc = 100 * running_acc / total
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

# 모델 학습 및 평가
for epoch in range(epochs):
    train(epoch)
    test(epoch)
    # path = os.path.join(save_path, f'epoch_{epoch}.pth')
    # torch.save(model.state_dict(), path)