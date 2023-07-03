# 필요한 라이브러리 import
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# import torchvision.models as models
from torch.utils.data import DataLoader
import dataset
import model

# 각종 path 및 파라미터 설정
data_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\ro_sci_pa'
save_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\model_para'
epochs = 30
batch_size = 32
learning_rate = 0.01

# gpu를 사용할 수 있으면 gpu를 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
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
num_val = int(num_data*0.2)
num_test = num_data - num_train - num_val

train_data, val_data, test_data = torch.utils.data.random_split(datasets, [num_train, num_val, num_test])

# dataloader 설정
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 모델, 손실함수, 옵티마이저 설정
teacher_model = model.ResNet18(num_classes=3)
teacher_model = teacher_model.to(device)
student_model = model.ResNet18(num_classes=3)
student_model = student_model.to(device)

teacher_model.load_state_dict(torch.load(os.path.join(save_path, f'teacher.pth')))

criterion = nn.CrossEntropyLoss()
dist_criterion = nn.CosineSimilarity(dim=-1)
optimizer = optim.SGD(student_model.parameters(), lr=learning_rate)

def train(epoch):
    print('\nEpoch: %d'%epoch)
    # student model train mode로 전환
    # teacher model은 eval mode
    teacher_model.eval()
    student_model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, t4, t3, t2, t1 = teacher_model(inputs)
        outputs, s4, s3, s2, s1 = student_model(inputs)
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        
        classif_loss = criterion(outputs, labels)
        distil_loss4 = torch.mean(dist_criterion(t4, s4))
        distil_loss3 = torch.mean(dist_criterion(t3, s3))
        distil_loss2 = torch.mean(dist_criterion(t2, s2))
        distil_loss1 = torch.mean(dist_criterion(t1, s1))
        
        distil_loss = distil_loss4 + distil_loss3 + distil_loss2 + distil_loss1
        loss = classif_loss + 0.2*distil_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    total_loss = running_loss / len(trainloader)
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

def test(epoch, loader, mode='val'):
    print('\nEpoch: %d'%epoch)
    # model eval mode로 전환
    teacher_model.eval()
    student_model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    global BEST_SCORE
    for (inputs, labels) in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        _, t4, t3, t2, t1 = teacher_model(inputs)
        outputs, s4, s3, s2, s1 = student_model(inputs)
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        
        classif_loss = criterion(outputs, labels)
        distil_loss4 = torch.mean(dist_criterion(t4, s4))
        distil_loss3 = torch.mean(dist_criterion(t3, s3))
        distil_loss2 = torch.mean(dist_criterion(t2, s2))
        distil_loss1 = torch.mean(dist_criterion(t1, s1))
        
        distil_loss = distil_loss4 + distil_loss3 + distil_loss2 + distil_loss1
        loss = classif_loss + 0.2*distil_loss

        running_loss += loss.item()
    total_loss = running_loss / len(loader)
    total_acc = 100 * running_acc / total
    if total_acc > BEST_SCORE and not mode=='test':
        path = os.path.join(save_path, f'student.pth')
        torch.save(student_model.state_dict(), path)
        BEST_SCORE = total_acc
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

# 모델 학습 및 평가
BEST_SCORE = 0
for epoch in range(epochs):
    train(epoch)
    test(epoch, valloader)

print("TEST-----------------------------------")
student_model.load_state_dict(torch.load(os.path.join(save_path, f'student.pth')))
test(-1, testloader, 'test')