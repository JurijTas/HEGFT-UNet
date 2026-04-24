#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:53:40 2026

@author: yurijtas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


    
# ----------------------------
# Simple ConvBlock
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                              padding=(kernel_size - 1)//2, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        assert x.size(1) == self.inp_dim, f"expected {self.inp_dim} channels, got {x.size(1)}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x
      

# -------------------------------------------------
# Multmodal Feature Fusion Block --- MFFB
# ------------------------------------------------
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class ChSAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.ConvBlock = nn.Conv2d(2,1,kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.ConvBlock(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.ConvBlock1 = ConvBlock(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.ConvBlock2 = ConvBlock(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.ConvBlock3 = ConvBlock(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim
    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x); out = self.relu(out)
        out = self.ConvBlock1(out); out = self.bn2(out); out = self.relu(out)
        out = self.ConvBlock2(out); out = self.bn3(out); out = self.relu(out)
        out = self.ConvBlock3(out)
        out += residual
        return out
    
    
class MFFB(nn.Module):
    def __init__(self, inch_tr, inch_cnn, outch, drop_rate=0.):
        super().__init__()
        self.inch1 = inch_tr
        # reduce/align channels from transformer -> inch_tr
        self.TrMaps = ConvBlock(inch_tr, inch_tr, 1, bn=True, relu=False)
        # map cnn channels to inch_tr so both branches have same channels
        self.CNNMaps = ConvBlock(inch_cnn, inch_tr, 1, bn=True, relu=False)

        self.FFN = FFN(inch_tr, inch_tr)
        self.ChSAM = ChSAMLayer(inch_tr)

        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # fuse two branches (each inch_tr), output outch (we will set outch = cnn_channels[i])
        self.residual = Residual(inch_tr + inch_tr, outch)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, t, cnn):
        inch1 = self.inch1
        # ensure matching H,W before fusion
        if t.shape[2:] != cnn.shape[2:]:
            cnn = F.interpolate(cnn, size=t.shape[2:], mode="bilinear", align_corners=False)

        # project / align channel dims
        TrMaps = self.TrMaps(t)        
        CNNMaps = self.CNNMaps(cnn)    

        # compute bi-linear style pooling for gating
        bp = self.avg_pool(TrMaps + CNNMaps).view(-1, inch1)   
        bp = self.softmax(self.FFN(bp)).view(-1, inch1, 1, 1)  

        # spatial + channel attention
        t = self.ChSAM(TrMaps)
        cnn = self.ChSAM(CNNMaps)

        # channel-wise gating and residual fusion
        fuse = self.residual(torch.cat([cnn * bp, t * bp], dim=1))

        return self.dropout(fuse) if self.drop_rate > 0 else fuse

