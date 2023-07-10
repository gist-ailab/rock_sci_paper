import cv2
import torch
import torch.nn
import model
import os
import time
import torchvision.transforms as transforms

save_path = '/home/ailab/Workspace/minhwan/rock_sci_paper/model_para'
# save_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\model_para'
rsp_model = model.ResNet18(num_classes=3)
rsp_model.load_state_dict(torch.load(os.path.join(save_path, f'teacher.pth')))
rsp_model = rsp_model.cuda()
label_dict = {0:'rock', 1:'scissors', 2:'paper'}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF
    
    if status:
        cv2.imshow('video', frame)
    
    if key == ord('p'):
        cv2.imwrite(os.path.join(save_path, 'inference.jpg'), frame)
        cv2.destroyAllWindows()
        break

img = cv2.imread(os.path.join(save_path, 'inference.jpg'))

img = transform(img)
img = img.unsqueeze(0).cuda()

rsp_model.eval()
output, _, _, _, _ = rsp_model(img)
_, pred = torch.max(output, 1)
label = label_dict[pred.item()]
if label is not None:
    print(label)
    print(output)