#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:06:23 2026

@author: yurijtas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#  -------------------------------------------
#  Edge Focused Feature Block 
# --------------------------------------------

class EFFB(nn.Module):
    """
    Edge Focused Feature Block : takes equal-channel CNN and Transformer features (same C,H,W)
    and returns concatenated refined features (channels doubled).
    """
    def __init__(self, in_chans_cnn: int, in_chans_tr: int):
        super().__init__()
        C = in_chans_cnn
        T = in_chans_tr
        # simple pathway for CNN
        self.cnn_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
        )
        # simple pathway for Transformer
        self.tr_conv = nn.Sequential(
            nn.Conv2d(T, T // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(T // 2),
            nn.ReLU(inplace=True),
        )
        # final merge -> produce same channels as original CNN skip if you prefer:
        self.out_conv = nn.Sequential(
            nn.Conv2d(C // 2 + C // 2, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        
        self.match_conv_out = nn.Sequential(
          nn.Conv2d(T//2,  C//2, kernel_size=1, bias=False),
          nn.BatchNorm2d(C//2),
          nn.ReLU(inplace=True)
         )
        

    def forward(self, cnn_feat: torch.Tensor, tr_feat: torch.Tensor) -> torch.Tensor:
        # ensure same spatial size
        if cnn_feat.shape[2:] != tr_feat.shape[2:]:
            tr_feat = F.interpolate(tr_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False)

        c = self.cnn_conv(cnn_feat)  
        t = self.tr_conv(tr_feat)    
        # match channels number for transforme and cnn T//2-> C//2
        if t.shape[1] != c.shape[1]:
            t = self.match_conv_out(t)
        
        cat = torch.cat([c, t], dim=1) 
        out = self.out_conv(cat)        
        return out
