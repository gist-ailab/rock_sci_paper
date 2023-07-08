# from glob import glob
# import shutil
# import os

# from_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\people\\6'
# to_path = 'C:\\Users\\USER\\Desktop\\GSH_CRP\\codes\\rock_sci_paper\\data\\ro_sci_pa_heo'

# gesture = {0:'rock', 1:'scissors', 2:'paper'}
# files_list = glob(from_path+'\\*')
# for i in range(len(files_list)):
#     file_path = files_list[i]
#     file_name = file_path.split('\\')[-1]
#     print(file_name)
#     name = file_name.split('.')[0]
#     if int(name) % 3 == 0:
#         shutil.copy(file_path, to_path+f'\\rock\\{int(name)//3}.jpg')
#     if int(name) % 3 == 1:
#         shutil.copy(file_path, to_path+f'\\scissors\\{int(name)//3}.jpg')
#     if int(name) % 3 == 2:
#         shutil.copy(file_path, to_path+f'\\paper\\{int(name)//3}.jpg')