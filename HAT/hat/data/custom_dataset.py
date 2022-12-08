
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

import skimage.color as sc

import time
from math import log10, sqrt
import zipfile

# 불러오는 데이터가 이미지 파일 형식인지 확인합니다.
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

# 본 베이스라인 코드에서는 이미지를 YCbCr 채널로 변경 후, Y채널만 초해상화합니다.
# 최신 방법들은 RGB 이미지를 그대로 사용하니, 해당 코드를 반드시 사용해야하는 것은 아닙니다.
def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

# 이미지 크롭 사이즈를 설정하는 함수입니다.
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# LR 이미지를 만드는 함수입니다.
# 본 베이스라인 코드에서는 HR 이미지에 해당 함수를 적용해서 x4 스케일만큼 다운샘플링한 이미지를 LR로 정의합니다.
# 초기부터 x4 스케일로 다운샘플링된 이미지 데이터셋도 제공합니다. 이 경우, 해당 함수는 필수가 아닙니다.
def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size//upscale_factor, Image.BICUBIC),
        ToTensor()
    ])

# HR 이미지를 만드는 함수입니다. 
# LR 이미지와 똑같은 크롭 사이즈를 사용해 자르고, 그것을 타겟 이미지로 설정합니다.
def target_transform(crop_size):
     return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])
     
# 앞서 정의한 함수들을 이용해 데이터셋을 생성하는 클래스를 정의합니다.
class DatasetFromFolder(data.Dataset):
    def __init__(self, hr_dir, lr_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.lr_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.lr_filenames[index])
        target = load_img(self.hr_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.hr_filenames)

# 64의 크롭 사이즈를 갖고 훈련 데이터셋을 만드는 함수입니다.
def get_training_set(hr_dir, lr_dir, upscale_factor):
   
    crop_size = calculate_valid_crop_size(64, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

# 64의 크롭 사이즈를 갖고 검증 데이터셋을 만드는 함수입니다.
# 함수명은 테스트(추론) 데이터셋으로 보일 수 있으나, 검증 데이터셋의 경우에도 test_set 등으로 명시하는 것이 일반적입니다.
def get_test_set(hr_dir, lr_dir, upscale_factor):
   
    crop_size = calculate_valid_crop_size(64, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))