# 필요한 라이브러리 import
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import rock_sci_paper.data.dataset as dataset

# 각종 path 및 파라미터 설정
data_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\dataset\\bench'
# save_path = '모델 파라미터를 저장할 디렉토리 경로'
epochs = 50
batch_size = 32
learning_rate = 0.01

# gpu를 사용할 수 있으면 gpu를 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform 설정
transform = nn.Sequential(
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
)

# dataset 설정
train_data = dataset.RockScissorPaper(
    root=data_path,
    train=True,
    download=True,
    transform=transform
)
test_data = dataset.RockScissorPaper(
    root=data_path,
    train=False,
    download=True,
    transform=transform
)

# dataloader 설정
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# model 설계
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(8*8*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        flatten = out.view(out.size(0), -1)
        score = self.fc(flatten)
        return score

# 모델, 손실함수, 옵티마이저 설정
# model = ConvNet()
model = models.resnet18(rpetrained=True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters, lr=learning_rate)

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