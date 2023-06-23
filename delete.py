import glob
import os

a = glob.glob("/home/ailab/Workspace/minhwan/highschool/data/*/*/*.jpg")

for i in a:
    os.remove(i)