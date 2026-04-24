#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:21:12 2025

@author: yurijtas
"""

import torch
from dataset import ThyroidDataset
from torch.utils.data import DataLoader
import numpy as np
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm
import cv2

def dice_score(preds, targets, smooth=1e-8):
    """
    Computes the Dice score for binary segmentation.
    
    Args:
        preds (torch.Tensor): Predicted masks with values between 0 and 1.
        targets (torch.Tensor): Ground truth masks with values 0 or 1.
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice score.
    """
    preds = ((preds) > 0.5).float()
    targets_mask=targets
    intersection = (preds * targets_mask).sum()
    union = preds.sum() + targets_mask.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = ThyroidDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = ThyroidDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader

###############################################################
def check_accuracy(loader, 
                      model, 
                      device,
                      dice_loss_fn,
                      bce_loss_fn,
                      boundary_loss_fn,
                      hd_loss_fn,
                      bce_weight = 0.5,
                      dice_weight = 0.5, 
                      boundary_weight = 0.0,
                      hd_weight = 0.0,
                      min_object_size=50,
                    ):
    total_loss = 0.0
    all_dice = []
    num_correct = 0
    num_pixels = 0
   
    model.eval()
    with torch.no_grad():
        for x, y ,sdfs in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()   # Ensure correct shape and type
            sdfs = sdfs.to(device).unsqueeze(1).float() 
            targets=y
        
            predictions = model(x)
              
            loss = (
                bce_weight * bce_loss_fn(predictions, targets)
                + dice_weight * dice_loss_fn(predictions, targets)
                + boundary_weight * boundary_loss_fn(predictions, sdfs)
                + hd_weight * hd_loss_fn(predictions, targets)
            )

            #  Compute loss (use .item() to avoid storing tensors)
            total_loss += loss.item()
            # Post-process predictions for metrics
            preds_np = torch.sigmoid(predictions).detach().cpu().numpy()  # (N,1,H,W)
            preds_np = preds_np.squeeze(1)  # (N,H,W)

            # Apply threshold and remove small objects
            processed_preds = []
            for p in preds_np:
                p_bin = adaptive_threshold(p, method=0.5)
                p_clean = remove_small_objects(p_bin, min_size=min_object_size)
                processed_preds.append(p_clean)
            processed_preds = np.stack(processed_preds)  # (N,H,W)

            # Compute metrics per image
            for i in range(processed_preds.shape[0]):
                pred_img = torch.tensor(processed_preds[i], device=device, dtype=torch.float32)
                target_img = y[i, 0]

                # Dice per image
                intersection = (pred_img * target_img).sum()
                dice = (2 * intersection + 1e-8) / (pred_img.sum() + target_img.sum() + 1e-8)
                all_dice.append(dice.item())

                # Pixel-wise accuracy
                num_correct += (pred_img == target_img).sum().item()
                num_pixels += torch.numel(pred_img)
                
                

    avg_loss = total_loss / len(loader)
    avg_dice = sum(all_dice) / len(all_dice)
    accuracy = 100.0 * num_correct / num_pixels
    print(f"\tAccuracy: {accuracy:.2f}%  |  Dice score (validation): {avg_dice:.4f}  |  Avg Loss (validation): {avg_loss:.4f}")
    model.train()
    return avg_dice, avg_loss, accuracy 


#####################################
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min', save_path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for score (e.g., Dice).
            save_path (str): Where to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
       
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == 'min':
            self.monitor_op = np.less
        else:  # 'max'
            self.monitor_op = np.greater

    def __call__(self, current_score, model, optimizer=None, epoch=None):
        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model, optimizer, epoch)
        # Check if improvement
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.save_checkpoint(model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            print(f" No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                

    def reset_counter(self):
       """Reset early stopping counter."""
       self.counter = 0
       self.early_stop =False
       
        
            
    def save_checkpoint(self, model, optimizer=None, epoch=None):
        """
        Saves model state_dict and architecture parameters.
        """
        # ---------------------------------------------
        # Construct checkpoint dict
        # ---------------------------------------------

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_params": getattr(model, "config", None),
            "dct_shapes_decoder": model.decoder.dct_shapes,
            "epoch": epoch
        }
          
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # ---------------------------------------------
        #  Save file
        # ---------------------------------------------
        torch.save(checkpoint, self.save_path)

        print(f"Improvement detected. Saving model to {self.save_path}")

       
        
    def load_checkpoint(self, model=None, optimizer=None, map_location="cuda"):
       """
       Loads the best model (and optionally optimizer) from checkpoint.
       Returns model parameters for automatic reconstruction if needed.
       """
       
       torch.serialization.add_safe_globals([ConfigDict]) ### 
       checkpoint = torch.load(self.save_path, map_location=map_location)
       
       # Return saved configuration and last epoch (useful for resuming)
       model_params = checkpoint.get('model_params', None)
       epoch = checkpoint.get('epoch', None)
     
       if model is not None:

         dct_shapes = checkpoint.get("dct_shapes_decoder", None)

         if dct_shapes is not None:
               
           for shape, fcsa in zip(dct_shapes, model.decoder.FCSatt_modules):
               if shape is None:
                   continue
               H, W = shape
               # initialize DCT layer(not built up during model construction)
               fcsa.frequency_channel._ensure_dct_layer(H, W)

      
         model.load_state_dict(checkpoint['model_state_dict'])
         print(f"\nLoaded model weights from checkpoint for epoch {epoch+1}.")

       # Option 2: optimizer is passed in, also restore its state
       if optimizer is not None and 'optimizer_state_dict' in checkpoint:
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          print(f"\nLoaded optimizer state from checkpoint for epoch {epoch+1}.")

      
       return model_params, epoch

def remove_small_objects(mask, min_size=50):
    """
    Remove small connected components from a binary mask.
    
    Args:
        mask: numpy array (H, W), binary (0/1)
        min_size: minimum number of pixels to keep a component
    
    Returns:
        cleaned_mask: numpy array (H, W), binary
    """
    # Ensure mask is uint8
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    # Create output mask
    cleaned_mask = np.zeros_like(mask_uint8)
    
    # Keep only components larger than min_size
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1
    
    return cleaned_mask

#### Adaptive thresholding   ###
'''
Usage: 
    pred_mask_bin = adaptive_threshold(pred_probs, method=0.5)
    pred_mask_clean = remove_small_objects(pred_mask_bin, min_size=100)

'''
def adaptive_threshold(prob_map, method="otsu"):
    """
    Convert probability map to binary mask.
    
    Args:
        prob_map: numpy array (H, W), float32
        method: "otsu" or float threshold
    Returns:
        binary mask (0/1)
    """
    if method == "otsu":
        prob_uint8 = (prob_map*255).astype(np.uint8)
        _, binary = cv2.threshold(prob_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = (prob_map > method).astype(np.uint8)
    return binary


##################################################################################################

def train_fn(epoch,
             num_epochs,
             loader, 
             model, 
             optimizer,
             scaler, 
             device,
             dice_loss_fn,
             bce_loss_fn,
             boundary_loss_fn,
             hd_loss_fn,
             bce_weight = 0.5,
             dice_weight = 0.5, 
             boundary_weight = 0.0,
             hd_weight = 0.0):
    
  loop = tqdm(loader)
  epoch_dice = 0.0
  epoch_loss = 0.0
 

  for batch_idx, (data, targets, sdfs) in enumerate(loop):
    loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")  
    
    data, targets, sdfs = data.to(device), targets.unsqueeze(1).to(device), sdfs.unsqueeze(1).to(device)
    # forward

   
    with torch.amp.autocast('cuda'):
     predictions = model(data)    
     loss = (
              bce_weight * bce_loss_fn(predictions, targets)
              + dice_weight * dice_loss_fn(predictions, targets)
              + boundary_weight * boundary_loss_fn(predictions, sdfs)
              + hd_weight * hd_loss_fn(predictions, targets)
     )

    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()    
    scaler.step(optimizer)
    scaler.update()
    #
    epoch_loss+=loss.item()
    # Compute Dice score
    dice = dice_score(torch.sigmoid(predictions), targets)
    epoch_dice += dice.item()
    # update tqdm loop
    
    loop.set_postfix({'loss': loss.item(), 'Dice': dice.item(), 'Batch': batch_idx})
  avg_dice = epoch_dice / len(loader)
  avg_loss = epoch_loss / len(loader)
  print(f"Epoch [{epoch+1}/{num_epochs}], Average Dice Score: {avg_dice:.4f}, Average Loss: {avg_loss:.4f}")
  return avg_loss, avg_dice

###############################################################
