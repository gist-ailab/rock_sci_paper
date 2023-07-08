import cv2
import time
import os
import argparse

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

parser = argparse.ArgumentParser(description='rock-scissor-paper-dataset')
parser.add_argument('--people_num', type=int)
parser.add_argument('--data_num', type=int, default=100)
args = parser.parse_args()

i=0
data_num = args.data_num
people_num = args.people_num

arr = []

for k in range(data_num):
    arr.append("rock_"+str(k))
    arr.append("scissors_"+str(k))
    arr.append("paper_"+str(k))

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
   
while webcam.isOpened():
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF
    
    if status:
        cv2.imshow(arr[i], frame)
        
    ### p 사진 찍기
    if key == ord('p'):
        if os.path.isfile("/home/ailab/Workspace/minhwan/rock_sci_paper/data/pic128/"+str(arr[i].split("_")[0])+"/"+str(int(arr[i].split("_")[1])+people_num*data_num)+".jpg"):
            print("Wrong people num!")
            break
        else:
            cv2.imwrite("/home/ailab/Workspace/minhwan/rock_sci_paper/data/pic128/"+str(arr[i].split("_")[0])+"/"+str(int(arr[i].split("_")[1])+people_num*data_num)+".jpg", frame)
            print("/home/ailab/Workspace/minhwan/rock_sci_paper/data/pic128/"+str(arr[i].split("_")[0])+"/"+str(int(arr[i].split("_")[1])+people_num*data_num)+".jpg")
            if not os.path.isdir("/home/ailab/Workspace/minhwan/rock_sci_paper/data/people/"+str(people_num)):
                os.mkdir("/home/ailab/Workspace/minhwan/rock_sci_paper/data/people/"+str(people_num))
            cv2.imwrite("/home/ailab/Workspace/minhwan/rock_sci_paper/data/people/"+str(people_num)+"/"+str(i)+".jpg", frame)
            
            i+=1
            cv2.destroyAllWindows()
            
    ### 잘못 찍었으면 이전 데이터 삭제하고 다시 시작
    if key == ord('q'):
        i-=1
        os.remove("data/pic128/"+str(arr[i].split("_")[0])+"/"+str(int(arr[i].split("_")[1])+people_num*data_num)+".jpg")
        os.remove("data/people/"+str(people_num)+"/"+str(i)+".jpg")
        cv2.destroyAllWindows()
        
    if i == len(arr):
        break

webcam.release()
