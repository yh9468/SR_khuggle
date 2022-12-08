import glob
import os

path = "./khuggle_train_clean_lr/*"
arr = glob.glob(path)

for i in range(len(arr)):
    new_name = arr[i].split('x')[0] + ".png"
    os.rename(arr[i], new_name)