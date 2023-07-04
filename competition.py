# 사람의 손동작을 읽어 그에 대응하여 이길 수 있는 제스쳐를 제시
import cv2
import torch
import torch.nn
import model
import os

save_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\model_para'

cap  = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

rsp_model = model.ResNet18(num_classes=3)
rsp_model.load_state_dict(torch.load(os.path.join(save_path, f'teacher.pth')))
rsp_model = rsp_model.cuda()
label_dict = {0:'rock', 1:'scissor', 2:'paper'}

while(True):
    ret, frame = cap.read()
    if (ret):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(frame, (256,256))
        img = torch.Tensor(img).unsqueeze(0).permute((0,3,1,2)).cuda()
        
        output, _, _, _, _ = rsp_model(img)
        _, pred = torch.max(output, 1)
        label = label_dict[pred.item()]
        print(label)
        cv2.putText(frame, "Yours : "+label, (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255))
        cv2.imshow('frame', frame)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
cap.release()