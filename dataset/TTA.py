import cv2
import glob
import os

arr = glob.glob("test/*")
os.makedirs("test_blur",exist_ok=True)
#os.makedirs("test_90")
#os.makedirs("test_180")
#os.makedirs("test_270")

for a in arr:
    name = os.path.basename(a)
    data = cv2.imread(a)
    #data = cv2.flip(data, 0)   # 1은 좌우반전
    blur2 = cv2.GaussianBlur(data,(3,3),0)       
    #img90 = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE) 
    #img180 = cv2.rotate(data, cv2.ROTATE_180) # 180도 회전
    #img270 = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
    #cv2.imwrite(os.path.join("test_90", name), img90)
    #cv2.imwrite(os.path.join("test_180", name), img180)
    #cv2.imwrite(os.path.join("test_270", name), img270)
    cv2.imwrite(os.path.join("test_blur", name), blur2)
    
