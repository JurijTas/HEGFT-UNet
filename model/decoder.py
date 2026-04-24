#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:09:50 2026

@author: yurijtas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List, Tuple
from .MFCSA import FcsAttention

#-----------------------------------------------------------------------
#    UNET Decoder with MFCSA modules embedded  on the left hand side only   
#--------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_ch:int, skip_ch:int, out_ch:int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UpsampleConvTranspose(nn.Module):
    def __init__(self, channels):
        super(UpsampleConvTranspose, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=2,   # 2x2 kernel
            stride=2         # stride=2 doubles spatial dimensions
        )

    def forward(self, x):
        return self.up(x)


class ChannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, c)
        x = self.linear(x)
        x = x.view(b, h, w, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
############################################
class UNetMFCSADecoder(nn.Module):
    """
    Expects:
      - skips: [s1, s2, s3, s4] where s1 -> 64, s2 -> 128, s3 -> 256, s4 -> 512
      - bottleneck channels: 1024
    Decoder path: 1024 -> 512 -> 256 -> 128 -> 64 -> final conv
    """
    def __init__(self, skip_channels:List[int], bottleneck_ch:int, num_classes:int=1,freq_sel_method = "top16"):
        super().__init__()
        assert len(skip_channels) == 4
        
        skip_channels_reverse = list(reversed(skip_channels)) 
        self.up_modules = nn.ModuleList()
        #
        self.freq_sel_method = freq_sel_method
        #
        for i in range(len(skip_channels)):
           in_ch = bottleneck_ch if i == 0 else skip_channels_reverse[i-1]
           out_ch = skip_channels_reverse[i]
           self.up_modules.append(UpBlock(in_ch, out_ch, out_ch))  
       

        ### FCSA modules - left branch          
        self.FCSatt_modules = nn.ModuleList()
        self.ChLin_modules = nn.ModuleList()
        for i in range(len(skip_channels)):
            in_ch = skip_channels_reverse[i] * 2 + skip_channels_reverse[i]
            out_ch = skip_channels_reverse[i]
            self.FCSatt_modules.append(FcsAttention(in_channels = in_ch, out_channels = in_ch, freq_sel_method=self.freq_sel_method))
            self.ChLin_modules.append(ChannelLinear(in_channels = in_ch, out_channels=out_ch))


        c1 = skip_channels_reverse[-1] # 64 skip connections of the first encoder layer
        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.num_classes = num_classes
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
            in_channels=c1,   # ← instead of dec_ch
            out_channels=c1,
            kernel_size=4, stride=2, padding=1  # 1/4 → 1/2
            ),
            nn.ConvTranspose2d(
            in_channels=c1,
            out_channels=c1,
            kernel_size=4, stride=2, padding=1  # 1/2 → 1/1
            )
        )

    def forward(self, bottleneck:torch.Tensor, skips:List[torch.Tensor], out_size:Tuple[int,int]=None):

        self.dct_shapes = []
        out = bottleneck

        for i, (up_m, fcsa_m, chlin_m) in enumerate(zip(self.up_modules, self.FCSatt_modules, self.ChLin_modules)):
            large = skips[-(i + 1)]

            if out.shape[2:] != large.shape[2:]:
                out_up = F.interpolate(out, size=large.shape[2:], mode='bilinear', align_corners=False)
            else:
                out_up = out
                
            
            fcsatt_input = torch.cat([out_up, large], dim=1)
            fcsatt_out = fcsa_m(fcsatt_input)
            fcsatt_out = chlin_m(fcsatt_out)

            out = up_m(out, fcsatt_out)
           
            self.dct_shapes.append(fcsa_m.frequency_channel.dct_hw)
            
           
        out = self.final_upsample(out)
        out = self.final_conv(out)    
      
        return out