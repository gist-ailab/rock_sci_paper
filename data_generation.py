import cv2
import time
import os
import argparse

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

parser = argparse.ArgumentParser(description='rock-scissor-paper-dataset')
parser.add_argument('--people', type=str)
parser.add_argument('--data_num', type=int, default=50)
args = parser.parse_args()

i=0
data_num = args.data_num
people = args.people

data_path = "C:\\Users\\minhwan\\rock_sci_paper\\data"

arr = []

for k in range(data_num):
    arr.append("rock_"+str(k))
    arr.append("scissors_"+str(k))
    arr.append("paper_"+str(k))

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
os.mkdir(data_path+"\\rock_scissor_paper")
os.mkdir(data_path+"\\rock_scissor_paper\\rock")
os.mkdir(data_path+"\\rock_scissor_paper\\scissors")
os.mkdir(data_path+"\\rock_scissor_paper\\paper")

while webcam.isOpened():
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF
    
    if status:
        cv2.imshow(arr[i], frame)
    
    ### p 사진 찍기
    if key == ord('p'):
        if os.path.isfile(data_path+"\\rock_scissor_paper\\"+str(arr[i].split("_")[0])+"\\"+people+str(int(arr[i].split("_")[1]))+".jpg"):
            print("Wrong people num!")
            break
        else:
            cv2.imwrite(data_path+"\\rock_scissor_paper\\"+str(arr[i].split("_")[0])+"\\"+people+str(int(arr[i].split("_")[1]))+".jpg", frame)
            print(data_path+"\\rock_scissor_paper\\"+str(arr[i].split("_")[0])+"\\"+people+str(int(arr[i].split("_")[1]))+".jpg")
            
            
            i+=1
            cv2.destroyAllWindows()
            
    ### 잘못 찍었으면 이전 데이터 삭제하고 다시 시작
    if key == ord('q'):
        i-=1
        os.remove(data_path+"\\rock_scissor_paper\\"+str(arr[i].split("_")[0])+"\\"+str(int(arr[i].split("_")[1])+people*data_num)+".jpg")
        cv2.destroyAllWindows()
        
    if i == len(arr):
        break

webcam.release()
