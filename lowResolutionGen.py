import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from glob import glob
import cv2
import numpy as np

gesture = 'paper'
d_path = f'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\ro_sci_pa\\{gesture}\\*'
save_path = f'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\LR_ro_sci_pa\\{gesture}\\'

img_paths = glob(d_path, recursive=True)
# print(img_paths)
for path in img_paths:
    name = path.split('\\')[-1]
    # print(path)
    img = cv2.imread(path)
    # print(img)
    img = F2.to_tensor(img)
    # img = img.permute((1,2,0))
    img = img.unsqueeze(0)
    h,w = img.shape[-2], img.shape[-1]
    img = F.interpolate(img, (h//8, w//8))
    img = F.interpolate(img, (h,w))
    img = img.squeeze()
    img = img.permute(1,2,0)
    img = img.numpy()
    cv2.imwrite(save_path+name, 255*img)