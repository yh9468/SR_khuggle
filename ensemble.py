import cv2
import glob
import numpy as np
import os

save_path = "final"
os.makedirs(save_path, exist_ok=True)

hat = glob.glob("./HAT/results/output/*")
swin = glob.glob("./SwinIR/results/output/*")

hat.sort()
swin.sort()

for i in range(len(hat)):
    name = os.path.basename(hat[i])
    
    hat_data = cv2.imread(hat[i])
    swin_data = cv2.imread(swin[i])
    
    hat_data = hat_data.astype(np.uint64)
    swin_data = swin_data.astype(np.uint64)
    
    origin_data = hat_data * 0.499 + swin_data * 0.501
    origin_data = origin_data.astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_path, name), origin_data)
    