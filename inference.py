from __future__ import print_function

import os
import os.path
from os.path import join
from os import listdir

import numpy as np
import random
from collections import OrderedDict
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


# 추론용 이미지(즉, 손상된 경희대 이미지)를 불러오고, 번호 순서대로 정렬합니다.
infer_dataset = "final"
infer_dataset_images = listdir(infer_dataset)
infer_dataset_images.sort()
np_list = []

for file in infer_dataset_images:
  out_np = np.asarray(Image.open(os.path.join(infer_dataset, file)))
  np_list.append(out_np)

# numpy array로 구성된 리스트를 numpy array 형태로 변환합니다.
np_submission_array = np.array(np_list)

# 'submission.npy' 형태로 최종 제출 파일을 정의합니다.
np.save("submission.npy", np_submission_array)
