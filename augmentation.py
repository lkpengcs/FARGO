import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_aug(m, s):
    process = A.Compose([
        A.Rotate(limit=(0, 360), p=0.8),

        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.8),

        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process


def train_crop_aug(m, s, size=192):
    process = A.Compose([
        A.OneOf((A.HorizontalFlip(p=0.8), A.VerticalFlip(p=0.8)), p=0.8),
        A.CenterCrop(size, size, 1),
        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process


def val_aug(m, s):
    process = A.Compose([
        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process


def val_crop_aug(m, s, size=400):
    process = A.Compose([
        A.CenterCrop(size, size, 1),
        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process
