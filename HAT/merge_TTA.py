import cv2
import glob
import numpy as np
import os

save_path = "results/output"
os.makedirs(save_path, exist_ok=True)

origin = glob.glob("./results/origin/*.png")
hor = glob.glob("./results/hor/*.png")
ver = glob.glob("./results/ver/*.png")
dir90 = glob.glob("./results/90/*.png")
dir180 = glob.glob("./results/180/*.png")
dir270 = glob.glob("./results/270/*.png")

origin.sort()
hor.sort()
ver.sort()
dir90.sort()
dir180.sort()
dir270.sort()

for i in range(len(origin)):
    name = os.path.basename(origin[i])
    
    origin_data = cv2.imread(origin[i])
    hor_data = cv2.imread(hor[i])
    ver_data = cv2.imread(ver[i])
    img90 = cv2.imread(dir90[i])
    img180 = cv2.imread(dir180[i])
    img270 = cv2.imread(dir270[i])
    
    hor_data = cv2.flip(hor_data, 1)
    ver_data = cv2.flip(ver_data, 0)
    img90 = cv2.rotate(img90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img180 = cv2.rotate(img180, cv2.ROTATE_180)
    img270 = cv2.rotate(img270, cv2.ROTATE_90_CLOCKWISE)
    
    origin_data = origin_data.astype(np.uint64)
    hor_data = hor_data.astype(np.uint64)
    ver_data = ver_data.astype(np.uint64)
    img90 = img90.astype(np.uint64)
    img180 = img180.astype(np.uint64)
    img270 = img270.astype(np.uint64)
    try:
        origin_data = origin_data + hor_data + ver_data + img90 + img180 + img270
    except:
        print(img90.shape)
        print(img180.shape)
        print(img270.shape)
        print(origin_data.shape)
        print(hor_data.shape)
        print(ver_data.shape)
        print(name)
        exit()
    origin_data = origin_data / 6
    origin_data = origin_data.astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_path, name), origin_data)
    