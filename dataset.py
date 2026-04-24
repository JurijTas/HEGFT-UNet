#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:23:16 2025

@author: yurijtas
"""
#import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_sdf(mask):
    """
    Compute Signed Distance Map (SDF) for a binary mask.
    Input: mask (H,W) with values {0,1}
    Output: float32 SDF in range approx [-maxdist, +maxdist]
    """
    mask = mask.astype(np.uint8)

    if mask.max() == 0:  
        # empty mask → SDF all positive distances
        sdf = distance_transform_edt(1-mask)
        return sdf.astype(np.float32)

    if mask.min() == 1:
        # full mask → SDF all negative distances
        sdf = -distance_transform_edt(mask)
        return sdf.astype(np.float32)

    # Foreground distance
    posdist = distance_transform_edt(mask)
    # Background distance
    negdist = distance_transform_edt(1 - mask)

    sdf = posdist - negdist
    return sdf.astype(np.float32)



class ThyroidBaseDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, return_paths=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_paths = return_paths

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        # Load image
        image = np.array(Image.open(img_path).convert("L"))
        image = np.expand_dims(image, axis=2)     # -> (H,W,1)

        # Load mask
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask /= mask.max() if mask.max() != 0 else 1
        mask = (mask > 0.5).astype(np.float32)    # binary mask (H,W)

        # Compute SDF
        sdf = compute_sdf(mask)                   # (H,W)

        # Apply Albumentations transforms
        
    
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask, sdf=sdf)
            image = augmented["image"]
            mask = augmented["mask"]
            sdf = augmented["sdf"]

        if self.return_paths:
            return image, mask, sdf, img_path, mask_path
        else:
            return image, mask, sdf


class ThyroidDataset(ThyroidBaseDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform, return_paths=False)


class ThyroidDatasetTL(ThyroidBaseDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform, return_paths=False)


class ThyroidTestDataset(ThyroidBaseDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform, return_paths=True)


