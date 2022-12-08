import cv2
import glob
import numpy as np
import os

save_path = "final"
origin = glob.glob("./test_results_x4/*")
hor = glob.glob("./test_hor_results_x4/*")
ver = glob.glob("./test_ver_results_x4/*")

origin.sort()
hor.sort()
ver.sort()
os.makedirs(save_path, exist_ok=True)

for i in range(len(origin)):
    name = os.path.basename(origin[i])
    origin_data = cv2.imread(origin[i])
    hor_data = cv2.imread(hor[i])
    ver_data = cv2.imread(ver[i])
    
    hor_data = cv2.flip(hor_data, 1)
    ver_data = cv2.flip(ver_data, 0)
    
    origin_data = origin_data.astype(np.uint64)
    hor_data = hor_data.astype(np.uint64)
    ver_data = ver_data.astype(np.uint64)
    
    origin_data = origin_data + hor_data + ver_data
    origin_data = origin_data / 3
    origin_data = origin_data.astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_path, name), origin_data)
    