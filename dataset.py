import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import PIL.Image as Image
import albumentations as A
import torch.nn.functional as F


class TrainDataSet(Dataset):
    def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
        self.transform = transform
        self.data_path = data_path
        self.mask_path = mask_path
        self.num_list = num_list
        self.class_num = class_num

        if self.class_num == 1:
            self.class_values = [0, 255]
        else:
            self.class_values = [0, 100, 255]

    def __getitem__(self, idx):
        if self.num_list is not None:
            images = cv2.imread(
                self.data_path + str(self.num_list[idx]) + '.bmp', 0)
            masks = cv2.imread(
                self.mask_path + str(self.num_list[idx]) + '.bmp', 0)
        else:
            images = cv2.imread(
                self.data_path + '.bmp')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            masks = cv2.imread(
                self.mask_path + '.bmp', 0)
        
        if self.class_num == 1:
            masks = np.where(masks == 100, 0, masks)
            masks = np.where(masks == 255, 1, masks)
        else:
            if self.class_values is not None:
                mask = [(masks == v) for v in self.class_values]
            masks = np.stack(mask, axis=-1).astype('float')

        if self.transform is not None:
            images = np.pad(images, 8, 'reflect')
            if self.class_num == 3:
                masks = np.pad(masks, ((8, 8), (8, 8), (0, 0)), 'reflect')
            elif self.class_num == 1:
                masks = np.pad(masks, 8, 'reflect')
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            if self.class_num == 3:
                masks = masks.transpose(0, 2)
                masks = masks.transpose(1, 2)
        return images, masks

    def __len__(self):
        return len(self.num_list)


class TestDataSet(Dataset):
    def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
        self.transform = transform
        self.data_path = data_path
        self.mask_path = mask_path
        self.num_list = num_list
        self.class_num = class_num

        if self.class_num == 1:
            self.class_values = [0, 255]
        else:
            self.class_values = [0, 100, 255]

    def __getitem__(self, idx):
        if self.num_list is not None:
            images = cv2.imread(
                self.data_path + str(self.num_list[idx]) + '.bmp', 0)
            masks = cv2.imread(
                self.mask_path + str(self.num_list[idx]) + '.bmp', 0)
        else:
            images = cv2.imread(
                self.data_path + '.bmp')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            masks = cv2.imread(
                self.mask_path + '.bmp', 0)
                
        if self.class_num == 1:
            masks = np.where(masks == 100, 0, masks)
            masks = np.where(masks == 255, 1, masks)
        else:
            if self.class_values is not None:
                mask = [(masks == v) for v in self.class_values]
            masks = np.stack(mask, axis=-1).astype('float')

        if self.transform is not None:
            images = np.pad(images, 8, 'reflect')
            if self.class_num == 3:
                masks = np.pad(masks, ((8, 8), (8, 8), (0, 0)), 'reflect')
            elif self.class_num == 1:
                masks = np.pad(masks, 8, 'reflect')
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            if self.class_num == 3:
                masks = masks.transpose(0, 2)
                masks = masks.transpose(1, 2)
        return images, masks

    def __len__(self):
        return len(self.num_list)