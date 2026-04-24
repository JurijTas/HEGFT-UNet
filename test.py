#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 08:06:33 2025

@author: yurijtas
"""

####  Here we load trained model and do testing usind data previously not used by the model
import torch, gc
import torchvision
from dataset import ThyroidTestDataset#CarvanaTestDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm ### needed for progress bar!
import metrics
import matplotlib
matplotlib.use('Agg') 
import cv2
import numpy as np
import os
import argparse

from ml_collections.config_dict import ConfigDict

from utils import (
    remove_small_objects,
    adaptive_threshold    
    )

from metrics import (
       compute_accuracy,
       compute_dice,
       compute_iou,
       compute_precision,
       compute_recall,
       compute_sensitivity,
       compute_specificity,
       compute_hausdorff,
       compute_hd95
    )
########################

####    Passing some arguments   #####
parser = argparse.ArgumentParser(description="Simple test script without main()")
parser.add_argument('--BCE_WEIGHT', type=float, default=.5, help="bce_weight in loss function combined of bce and dice")
parser.add_argument('--DICE_WEIGHT', type=float, default=.5, help="dice_weight in loss function combined of bce and dice")
parser.add_argument('--BOUNDARY_WEIGHT', type=float, default=.1, help="boundary_weight in loss function combined of bce and dice")
parser.add_argument('--HD_WEIGHT', type=float, default=.1, help="hd_weight in loss function combined of bce and dice")
parser.add_argument('--MODELS_DIR', type=str, help="folder where thw model to be tested was saved during trainig")
parser.add_argument('--SAVED_MODEL',type=str,help='basic model name')
parser.add_argument('--IMAGE_HEIGHT', type=int,default=320,help="Image height")
parser.add_argument('--IMAGE_WIDTH', type=int,default=256,help="Image width")
parser.add_argument('--TEST_IMG_DIR',type=str,help="Path to training image dataset folder")
parser.add_argument('--TEST_MASK_DIR',type=str,help="Path to training mask dataset folder")

args = parser.parse_args()


#######################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 50
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL = True
BCE_WEIGHT=args.BCE_WEIGHT
DICE_WEIGHT=args.DICE_WEIGHT
HD_WEIGHT=args.HD_WEIGHT
BOUNDARY_WEIGHT=args.BOUNDARY_WEIGHT
MIN_OBJ_SIZE = 50


IMAGE_HEIGHT=args.IMAGE_HEIGHT
IMAGE_WIDTH=args.IMAGE_WIDTH
#
MODELS_DIR=args.MODELS_DIR
SAVED_MODEL=args.SAVED_MODEL


'''
    Test basic model on chinese data set  --- for publication    
    Same CHIENSE data sets as used for training --- good for publication
'''


TEST_IMG_DIR = args.TEST_IMG_DIR
TEST_MASK_DIR = args.TEST_MASK_DIR


##
### folder where predictions are saved
TEST_MASK_DIR_PREDS_IPPT = "TEST/saved_masks_IPPT/preds/"
TEST_MASK_DIR_TARGETS_IPPT = "TEST/saved_masks_IPPT/targets/"

TEST_MASK_DIR_PREDS_TN3K= "TEST/saved_masks_TN3K/preds/"
TEST_MASK_DIR_TARGETS_TN3K = "TEST/saved_masks_TN3K/targets/"


DATA_DIR_TN3K = "TEST/saved_images_TN3K/"
DATA_DIR_IPPT = "TEST/saved_images_IPPT/"

FIGS ="TEST/figs_test/"
DATA="TEST/data/"




if not os.path.exists(TEST_MASK_DIR_PREDS_IPPT):
    os.makedirs(TEST_MASK_DIR_PREDS_IPPT)
    print("folder cerated: ", TEST_MASK_DIR_PREDS_IPPT)
    

if not os.path.exists(TEST_MASK_DIR_TARGETS_IPPT):
    os.makedirs(TEST_MASK_DIR_TARGETS_IPPT)
    print("folder cerated: " ,TEST_MASK_DIR_TARGETS_IPPT)
    
    
if not os.path.exists(TEST_MASK_DIR_PREDS_TN3K):
    os.makedirs(TEST_MASK_DIR_PREDS_TN3K)
    print("folder cerated: ", TEST_MASK_DIR_PREDS_TN3K)
    

if not os.path.exists(TEST_MASK_DIR_TARGETS_TN3K):
    os.makedirs(TEST_MASK_DIR_TARGETS_TN3K)
    print("folder cerated: " ,TEST_MASK_DIR_TARGETS_TN3K)    
    
if not os.path.exists(DATA_DIR_TN3K):
    os.makedirs(DATA_DIR_TN3K)
    print("folder cerated: ", DATA_DIR_TN3K)
    

if not os.path.exists(DATA_DIR_IPPT):
    os.makedirs(DATA_DIR_IPPT)
    print("folder cerated: " , DATA_DIR_IPPT)    
       
    
    
    
    
    
if not os.path.exists(FIGS):
   os.makedirs(FIGS)
   print("folder created: ", FIGS)

if not os.path.exists(DATA):
    os.makedirs(DATA)
    print("folder created: ",DATA)
    

##### folder where feates from last layer are stored for further analysis in MATLAB
#FEATURE_MAPS_DIR="feature_maps/" 




test_transform = A.Compose(
     [
         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Normalize(
             #mean=[0.0, 0.0, 0.0],
             #std=[1.0, 1.0, 1.0],
             mean = (0.0,),
             std = (1.0,), 
             max_pixel_value=255.0,
         ),
         ToTensorV2(),
     ],
     additional_targets={"sdf": "mask"}
     #additional_targets={"sdf": "image"}
)



test_ds = ThyroidTestDataset(
    image_dir=TEST_IMG_DIR,
    mask_dir=TEST_MASK_DIR,
    transform=test_transform,
    
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False,
)

##########################################

model_path = os.path.join(MODELS_DIR,SAVED_MODEL)

#########################################

#Loading the model later (for inference)

# Load checkpoint safely


torch.serialization.add_safe_globals([ConfigDict])

checkpoint = torch.load(model_path, map_location="cuda")
model_params = checkpoint.get('model_params', None)
dct_shapes = checkpoint.get("dct_shapes_decoder", None)

# Restore architecture from saved config

if "model_params" in checkpoint and checkpoint["model_params"] is not None:
   print("Reconstructing model with parameters:")
   for k, v in model_params.items():
       print(f"   {k:<25}: {v}")
else:
   print("Reconstructing model with default parameters:")


from model.model import HEGFTUNeT
model =  HEGFTUNeT(**model_params)

    
model_name=f"{model.__class__.__name__}"
print(f"model {model_name} has been initialized ")  
    
if dct_shapes is not None:
    for shape, fcsa in zip(dct_shapes, model.decoder.FCSatt_modules):
        if shape is None:
            continue
        H, W = shape
        # initialize DCT layer(not built up during model construction)
        fcsa.frequency_channel._ensure_dct_layer(H, W)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device=DEVICE)
model.eval()   


#######################################################################
dice_loss_fn = metrics.DiceLoss()
bce_loss_fn = metrics.BCEWithLogitsLoss()
boundary_loss_fn = metrics.BoundaryLoss()
hd_loss_fn = metrics.HausdorffDistanceLoss()
#####################################

hd95_test = 0
num_correct = 0
num_pixels = 0
dice_score_test = 0
loss_test = 0
iou_test = 0
precision_test=0
recall_test=0
sensitivity_test=0
specificity_test=0
hausdorff_test = 0
empty_preds=0

hd95_for_example=[]
precision_for_example=[]
recall_for_example=[]
sensitivity_for_example=[]
specificity_for_example=[]
hausdorff_for_example=[]
acc_for_example = []
dice_score_for_example=[]
iou_for_example = []
example_count = []
image_names = []
mask_names = []
f1score_for_example = []

loop= tqdm(test_loader,total=len(test_loader), desc="Testing")
with torch.no_grad():
  for idx, (data, targets, sdfs, data_pth, target_pth) in enumerate(loop):
    example_count.append(idx+1)
    image_names.append(data_pth)
    mask_names.append(target_pth)
    data = data.to(device=DEVICE)
    targets = targets.to(DEVICE).unsqueeze(1)
    sdfs = sdfs.to(DEVICE).unsqueeze(1)
    y=targets
   
    predictions = model(data)
    # Loss
    loss = (
           BCE_WEIGHT* bce_loss_fn(predictions, targets)
           + DICE_WEIGHT * dice_loss_fn(predictions, targets)
           + BOUNDARY_WEIGHT * boundary_loss_fn(predictions, sdfs)
           + HD_WEIGHT * hd_loss_fn(predictions, targets)
    )

    loss_test +=loss.item()
    
    preds = torch.sigmoid(predictions).detach().cpu().numpy()  # (N,1,H,W)
    preds = preds.squeeze(1)  # (N,H,W)

    # Apply threshold and remove small objects
    processed_preds = []
    for p in preds:
        p_bin = adaptive_threshold(p, method=0.5)
        p_clean = remove_small_objects(p_bin, min_size=MIN_OBJ_SIZE)
        processed_preds.append(p_clean)
    processed_preds = np.stack(processed_preds)  # (N,H,W)

    # Compute metrics per image
    for i in range(processed_preds.shape[0]):
  
        pred_img = torch.from_numpy(processed_preds[i]).float()
        target_img = targets[i, 0]
        data_img = data[i,0]
        data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min() + 1e-8)
        
        pred_img = pred_img.cpu()
        target_img = target_img.cpu()     

        # Dice per image
        dscore = float(compute_dice(pred_img, target_img))
        dice_score_for_example.append(dscore)
        
        #IoU per image
        
        iou = float(compute_iou(pred_img, target_img))
        iou_for_example.append(iou)
        
        # Pixel-wise accuracy
        # Accuracy
        correct, total = compute_accuracy(pred_img, target_img)
        correct = float(correct)
        total = float(total)

        acc_for_example.append(correct/total*100)
        ###### Other metrics  #######
        if pred_img.sum() == 0 or target_img.sum() == 0:
          hd95_val = float('nan')
          hausdorff_val = float('nan')
          empty_preds += 1
        else:
          hd95_val = compute_hd95(pred_img, target_img)   # tensor
          hd95_val = hd95_val.mean().item()               # convert to float

          hausdorff_val = compute_hausdorff(pred_img, target_img)
          hausdorff_val = hausdorff_val.mean().item()     # convert to float

        hd95_for_example.append(hd95_val)
        hausdorff_for_example.append(hausdorff_val)   
        ###
        precision = float(compute_precision(pred_img, target_img))
        precision_for_example.append(precision)
        ###
        recall = float(compute_recall(pred_img, target_img))
        recall_for_example.append(recall)
        ####
        sensitivity = float(compute_sensitivity(pred_img, target_img))
        sensitivity_for_example.append(sensitivity)
        ####
        specificity = float(compute_specificity(pred_img, target_img))
        specificity_for_example.append(specificity)
        ####
        f1score_for_example.append(2*precision*recall/(precision+recall+1e-8))
        
        global_idx = idx * test_loader.batch_size + i
        torchvision.utils.save_image(data_img.unsqueeze(0), f"{DATA_DIR_TN3K}/image_{global_idx}.png")
        
        cv2.imwrite(f"{TEST_MASK_DIR_PREDS_TN3K}/pred_{global_idx}.png", (pred_img.numpy()*255).astype(np.uint8))
        cv2.imwrite(f"{TEST_MASK_DIR_TARGETS_TN3K}/target_{global_idx}.png", (target_img.numpy()*255).astype(np.uint8))
       
    
print(f"Model generated {empty_preds} empty predictions")


ACC_test_mean = np.nanmean(acc_for_example)
ACC_test_std  = np.nanstd(acc_for_example)
dice_score_test_mean = np.nanmean(dice_score_for_example)
dice_score_test_std  = np.nanstd(dice_score_for_example)
hd95_test_mean = np.nanmean(hd95_for_example)
hd95_test_std  = np.nanstd(hd95_for_example)
hausdorff_test_mean = np.nanmean(hausdorff_for_example)
hausdorff_test_std  = np.nanstd(hausdorff_for_example)
iou_test_mean = np.nanmean(iou_for_example)
iou_test_std = np.nanstd(iou_for_example)
precision_test_mean = np.nanmean(precision_for_example)
precision_test_std = np.nanstd(precision_for_example)
recall_test_mean = np.nanmean(recall_for_example)
recall_test_std =np.nanstd(recall_for_example)
sensitivity_test_mean=np.nanmean(sensitivity_for_example)
sensitivity_test_std=np.nanstd(sensitivity_for_example)
specificity_test_mean=np.nanmean(specificity_for_example)
specificity_test_std=np.nanstd(specificity_for_example)

f1score_test_mean = np.nanmean(f1score_for_example)
f1score_test_std = np.nanstd(f1score_for_example)


####  
print("<<<  Test results for HEGFTUNeT model ")
print(f"\n\tDice score => mean {dice_score_test_mean:.4f} | std {dice_score_test_std:.4f}")
print(f"\tIoU => mean {iou_test_mean:.4f} | std {iou_test_std:.4f}")
print(f"\tAverage Acc => mean {ACC_test_mean:.2f}% | std {ACC_test_std:.2f}%")
#
print(f"\tRecall => mean {recall_test_mean:.4f} | std {recall_test_std:.4f}")
print(f"\tPrecision => mean {precision_test_mean:.4f} | std {precision_test_std:.4f}")
print(f"\tHD95 =>  mean {hd95_test_mean:.4f} | std {hd95_test_std:.4f}")

print(f"\tSensitivity => mean {sensitivity_test_mean:.4f} | std {sensitivity_test_std:.4f}")
print(f"\tSpecificity => mean {specificity_test_mean:.4f} | std {specificity_test_std:.4f}")
print(f"\tHausdorff => mean {hausdorff_test_mean:.4f} | std {hausdorff_test_std:.4f}")


import pandas as pd


# Convert metric lists directly to NumPy arrays
DC_tensor   = np.array(dice_score_for_example, dtype=float)
IoU_tensor  = np.array(iou_for_example, dtype=float)
ACC_tensor  = np.array(acc_for_example, dtype=float)
HD95_tensor = np.array(hd95_for_example, dtype=float)
PREC_tensor = np.array(precision_for_example, dtype=float)
REC_tensor  = np.array(recall_for_example, dtype=float)
SENS_tensor = np.array(sensitivity_for_example, dtype=float)
SPEC_tensor = np.array(specificity_for_example, dtype=float)
HD_tensor   = np.array(hausdorff_for_example, dtype=float)
F1_tensor = np.array(f1score_for_example, dtype=float)

EX_n = example_count

# Build DataFrame
DC_frame = pd.DataFrame({
    'Example number': EX_n,
    'Dice score': DC_tensor,
    'F1_score' : F1_tensor,
    'ACC': ACC_tensor,
    'IoU': IoU_tensor,
    'HD95': HD95_tensor,
    'Precision': PREC_tensor,
    'Recall': REC_tensor,
    'Sensitivity': SENS_tensor,
    'Specificity': SPEC_tensor,
    'Hausdorff': HD_tensor,
    'Image': image_names,
    'Mask': mask_names
})

# Save table
DC_frame.to_csv(f"{DATA}Test_Metrics.csv", index=False)


#####  Free all GPU memory  ###

del model
#del optimizer
torch.cuda.empty_cache()
gc.collect()
