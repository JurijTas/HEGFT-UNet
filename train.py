#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:41:32 2025

@author: yurijtas
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import os
import metrics
import matplotlib
matplotlib.use('Agg') 
import argparse

def str_or_none(value):
    '''
    Convert 'None' string to Python None, leave other strings as-is
    '''
    if value.lower()=="none":
        return None
    else:
        return value

####    Passing some arguments   #####
parser = argparse.ArgumentParser(description="Simple test script without main()")
parser.add_argument('--BCE_WEIGHT', type=float, default=.5, help="bce_weight in loss function combined of bce and dice")
parser.add_argument('--DICE_WEIGHT', type=float, default=.5, help="dice_weight in loss function combined of bce and dice")
parser.add_argument('--BOUNDARY_WEIGHT', type=float, default=.1, help="boundary_weight in loss function combined of bce and dice")
parser.add_argument('--HD_WEIGHT', type=float, default=.1, help="hd_weight in loss function combined of bce and dice")
parser.add_argument('--MODELS_DIR', type=str, help="folder where model to be tested was saved during trainig")
parser.add_argument('--SAVED_MODEL',type=str,help="basic model name")
parser.add_argument('--EARLY_STOPPER_MODE', type=str, default ='min',help="if min - early stopper looks at loss, if max - it looks at dice ora accyracy")
parser.add_argument('--EMBED_DIM', type=int,default=64, help ="imbedded dimension in Transformer Encoder Block")
parser.add_argument('--PATCH_SIZE',type=int,default=8,help="patch size for image tokenization in Transformer Encoder Block"),
parser.add_argument('--CNN_WEIGHTS', type=str_or_none,default=None,help="weights for initialization of cnn encoder : None (default), or 'imagenet'")
parser.add_argument('--IMAGE_HEIGHT', type=int,default=320,help="Image height")
parser.add_argument('--IMAGE_WIDTH', type=int,default=256,help="Image width")
parser.add_argument('--CNN_ENCODER', type=str,default="resnet34",help="default cnn encoder backbone")
parser.add_argument('--TRAIN_IMG_DIR',type=str,help="Path to training image dataset folder")
parser.add_argument('--TRAIN_MASK_DIR',type=str,help="Path to training mask dataset folder")
parser.add_argument('--VAL_IMG_DIR',type=str,help="Path to validation image dataset folder")
parser.add_argument('--VAL_MASK_DIR',type=str,help="Path to validation mask dataset folder")
args = parser.parse_args()
############################################

from utils import(
    get_loaders,
    check_accuracy,
    EarlyStopping,
    train_fn
)

# Hyperparameters etc.
LEARNING_RATE=1E-4;
WEIGHT_DECAY=1E-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS =100
NUM_WORKERS = 0#2
PIN_MEMORY = False#True
LOAD_MODEL = False
## Early stopper params
PATIENCE = 15
MIN_DELTA=0 #  1
SINGLE_LOSS=1

BCE_WEIGHT=args.BCE_WEIGHT
DICE_WEIGHT=args.DICE_WEIGHT
HD_WEIGHT=args.HD_WEIGHT
BOUNDARY_WEIGHT=args.BOUNDARY_WEIGHT
EARLY_STOPPER_MODE = args.EARLY_STOPPER_MODE


##############################
IMAGE_HEIGHT=args.IMAGE_HEIGHT
IMAGE_WIDTH=args.IMAGE_WIDTH
#
HEIGHT_MAX=2*IMAGE_HEIGHT
WIDTH_MAX=2*IMAGE_WIDTH
###############################
MODELS_DIR=args.MODELS_DIR
SAVED_MODEL=args.SAVED_MODEL
EMBED_DIM=args.EMBED_DIM
PATCH_SIZE=args.PATCH_SIZE
CNN_ENCODER=args.CNN_ENCODER
MIN_OBJ_SIZE=50
######
TRAIN_IMG_DIR = args.TRAIN_IMG_DIR 
TRAIN_MASK_DIR = args.TRAIN_MASK_DIR
VAL_IMG_DIR = args.VAL_IMG_DIR
VAL_MASK_DIR = args.VAL_MASK_DIR

###############################
CNN_WEIGHTS=args.CNN_WEIGHTS

#############################################

#
def main():
  #### Transformation with rescaling ###
  
  train_transform = A.Compose(
      [
          A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
          A.Rotate(limit=35, p=1.0),
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.1),
          A.Normalize(        
              mean=(0.0,),
              std=(1.0,),
              max_pixel_value=255.0,
          ),
          ToTensorV2(),
      ],
      additional_targets={"sdf": "mask"}
  )
  val_transforms = A.Compose(
      [
          A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
          A.Normalize(
              mean=(0.0,),
              std=(1.0,),
              max_pixel_value=255.0,
          ),
          ToTensorV2(),
      ],
      additional_targets={"sdf": "mask"}
  )

  torch.cuda.empty_cache()


     
  from model.model import HEGFTUNeT
  model = HEGFTUNeT(cnn_encoder_name=CNN_ENCODER, cnn_encoder_weights=CNN_WEIGHTS,
                           #cnn_channels = [64, 128, 256, 512],
                           transformer_depths=(1,1,1,1), #default (1,1,1,1)
                           transformer_heads=(2,4,8,16), #default = (2,4,8,16)
                           patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_classes=1,
                           trans_channels = [EMBED_DIM, 2*EMBED_DIM , 4*EMBED_DIM, 8*EMBED_DIM],
                           use_jt_fusion_block=True#,True,
  ).to(device = DEVICE, dtype=torch.float) 
  
  

  model_path = os.path.join(MODELS_DIR,SAVED_MODEL)
  if not os.path.exists(MODELS_DIR):
      os.makedirs(MODELS_DIR)
      print("folder cerated: " , MODELS_DIR)    
  
  
  

  dice_loss_fn = metrics.DiceLoss()
  bce_loss_fn = metrics.BCEWithLogitsLoss()
  boundary_loss_fn = metrics.BoundaryLoss()
  hd_loss_fn = metrics.HausdorffDistanceLoss()  
      

          
  optimizer = optim.AdamW(
      model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  train_loader, val_loader = get_loaders(
          TRAIN_IMG_DIR,
          TRAIN_MASK_DIR,
          VAL_IMG_DIR,
          VAL_MASK_DIR,
          BATCH_SIZE,
          train_transform,
          val_transforms,
          NUM_WORKERS,
          PIN_MEMORY,
  )
    
  scaler = torch.amp.GradScaler('cuda')
  
#  if LOAD_MODEL:
#    model.load_state_dict(torch.load(model_path))
    
  epoch_count = []
  train_dice_values = []
  train_loss_values = []
  test_dice_values = []
  test_loss_values = []
  test_acc_values = []
      
  early_stopper = EarlyStopping(
     patience=PATIENCE, min_delta=MIN_DELTA, mode=EARLY_STOPPER_MODE, save_path=model_path)
      
  
      
   
  for epoch in range(NUM_EPOCHS):
     epoch_count.append(epoch)   
             
     print(f"Training with Loss = {BCE_WEIGHT}BCE + {DICE_WEIGHT}Dice")
             
     train_loss,train_dice = train_fn(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            hd_loss_fn=hd_loss_fn,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            boundary_weight=BOUNDARY_WEIGHT,
            hd_weight=HD_WEIGHT
    )
          
          
     train_loss_values.append(train_loss)
     train_dice_values.append(train_dice)

    # Check accuarcy
        
     test_dice, test_loss, test_acc = check_accuracy(
            loader=val_loader,
            model=model,
            device=DEVICE,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            hd_loss_fn=hd_loss_fn,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            boundary_weight=BOUNDARY_WEIGHT,
            hd_weight=HD_WEIGHT,
            min_object_size=MIN_OBJ_SIZE
     )
    
     test_dice_values.append(test_dice)
     test_loss_values.append(test_loss)
     test_acc_values.append(test_acc)
 
     if EARLY_STOPPER_MODE == 'min':
        early_stopper(test_loss, model, epoch=epoch)
     else:
        early_stopper(test_dice, model, epoch=epoch)
    
     if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}!")
        early_stopper.reset_counter()
        break
    
     # After basic training finishes, restore the best model
     early_stopper.load_checkpoint(model=model)
        
  

  torch.cuda.empty_cache()
  del model, optimizer, bce_loss_fn, dice_loss_fn, boundary_loss_fn, hd_loss_fn
  
  
if __name__ == "__main__":
  main()



  


