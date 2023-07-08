import os
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from glob import glob
import cv2
import numpy as np
import random

seed = 2023
tr_rate = 0.8
va_rate = 0.1
te_rate = 1-tr_rate-va_rate


np.random.seed(seed)
random.seed(seed)

save_path = f'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\ro_sci_pa_heo\\'

tr_f = open(os.path.join(save_path, 'train.txt'), 'w')
va_f = open(os.path.join(save_path, 'val.txt'), 'w')
te_f = open(os.path.join(save_path, 'test.txt'), 'w')

total_names = []

gestures = ['paper', 'rock', 'scissors']
for gesture in gestures:
    # gesture = 'scissor'
    d_path = f'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\ro_sci_pa_heo\\{gesture}\\*'

    img_paths = glob(d_path, recursive=True)
    # print(img_paths)
    for path in img_paths:
        name = path.split('\\')[-1]
        total_names.append(f'{gesture}\\{name}')
        
random.shuffle(total_names)

tr_list = total_names[:int(len(total_names)*tr_rate)]
va_list = total_names[int(len(total_names)*tr_rate):int(len(total_names)*(tr_rate+va_rate))]
te_list = total_names[int(len(total_names)*(tr_rate+va_rate)):]

for name in tr_list:
    tr_f.write(name+'\n')
for name in va_list:
    va_f.write(name+'\n')
for name in te_list:
    te_f.write(name+'\n')
    
tr_f.close()
va_f.close()
te_f.close()