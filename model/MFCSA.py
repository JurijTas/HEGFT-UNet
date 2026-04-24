#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:47:15 2025

@author: yurijtas
"""
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
###----------------------------------------------
###                MFCSA  Module              ####
###----------------------------------------------
# keep your original frequency index helper
def get_freq_indices(method: str):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


####################################################################################
# Spatial attention (unchanged)
####################################################################################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        y = self.conv1(cat)
        return self.sigmoid(y)  # [B,1,H,W]


####################################################################################
# DCT filter generator (works for arbitrary HxW and mapper lists)
####################################################################################
class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height: int, width: int, mapper_x: List[int], mapper_y: List[int], channel: int):
        """
        Build DCT filter bank for given tile size (height,width), mapper indices and channel dim.
        The returned buffer 'weight' has shape [channel, height, width].
        Channel partitions are distributed evenly; remainder goes to last partition.
        """
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        self.num_freq = len(mapper_x)

        # build dct filters and register as buffer
        weight = self.get_dct_filter(height, width, mapper_x, mapper_y, channel)
        self.register_buffer('weight', weight)  # fixed spectral filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or pooled [B,C,H',W'] matching weight's spatial dims
        # multiply by DCT weights and sum spatially
        assert x.dim() == 4, f'x must be 4D, got {x.dim()}'
        # weight [C, Hk, Wk], x [B,C,Hk,Wk]
        self.weight = self.weight.to(x.device)  # ensure same device
        x_weighted = x * self.weight  # broadcasts per-channel DCT weights
        result = torch.sum(x_weighted, dim=[2, 3])  # [B, C]
        return result  # [B, C]

    def build_filter(self, pos: int, freq: int, P: int) -> float:
        # standard DCT basis value
        res = math.cos(math.pi * freq * (pos + 0.5) / P) / math.sqrt(P)
        if freq == 0:
            return res
        else:
            return res * math.sqrt(2.0)

    def get_dct_filter(self, tile_h: int, tile_w: int, mapper_x: List[int], mapper_y: List[int], channel: int) -> torch.Tensor:
        # allocate
        dct_filter = torch.zeros(channel, tile_h, tile_w, dtype=torch.float32)

        num_freq = len(mapper_x)
        base = channel // num_freq
        remainder = channel - base * num_freq

        # For each chosen frequency (u_x, v_y), fill c_part channels
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            # number of channels assigned to this frequency
            c_part = base + (1 if i == (num_freq - 1) and remainder > 0 else 0)
            ch_start = i * base
            ch_end = ch_start + c_part
            # handle last chunk index correction if remainder > 0
            if i == (num_freq - 1):
                ch_end = channel
                ch_start = channel - c_part

            # compute 2D basis
            for th in range(tile_h):
                bx = self.build_filter(th, u_x, tile_h)
                for tw in range(tile_w):
                    by = self.build_filter(tw, v_y, tile_w)
                    dct_filter[ch_start:ch_end, th, tw] = bx * by

        return dct_filter  # [C, H, W]


####################################################################################
# MultiSpectralAttentionLayer - builds DCT layer lazily and handles arbitrary HxW
####################################################################################
class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16, freq_sel_method: str = 'top16', max_dct_size: int = None):
        """
        channel: input channels
        reduction: reduction ratio for FC
        freq_sel_method: one of get_freq_indices keys (e.g. 'top16')
        max_dct_size: optional cap for DCT spatial size to limit memory (e.g. 64). If None -> use pooled size = input size.
        """
        super(MultiSpectralAttentionLayer, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.freq_sel_method = freq_sel_method
        self.max_dct_size = max_dct_size  # optional cap to avoid giant DCT filters

        # base mapper in 7x7 reference space
        base_mapper_x, base_mapper_y = get_freq_indices(freq_sel_method)
        self.base_mapper_x = base_mapper_x
        self.base_mapper_y = base_mapper_y
        self.num_split = len(base_mapper_x)

        # fc which maps channel -> channel attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # DCT layer will be created lazily when we know spatial dims
        self.dct_layer = None
        self.dct_hw = None  # tuple (h,w) of dct_layer
        

    def _scale_mapper(self, mapper_x: List[int], mapper_y: List[int], Hk: int, Wk: int) -> Tuple[List[int], List[int]]:
        """
        Scale indices from 7x7 reference to (Hk, Wk) grid.
        Use rounding and clamp to valid range.
        """
        scaled_x = []
        scaled_y = []
        for mx, my in zip(mapper_x, mapper_y):
            # factor = target / 7.0
            fx = float(Hk) / 7.0
            fy = float(Wk) / 7.0
            sx = int(round(mx * fx))
            sy = int(round(my * fy))
            sx = max(0, min(Hk - 1, sx))
            sy = max(0, min(Wk - 1, sy))
            scaled_x.append(sx)
            scaled_y.append(sy)
        return scaled_x, scaled_y

    def _ensure_dct_layer(self, Hk: int, Wk: int):
        """
        Create or recreate the MultiSpectralDCTLayer for the given Hk, Wk
        if it doesn't exist or the size changed.
        """
        # optionally cap Hk/Wk to avoid huge DCT buffers
        cap_h = Hk if self.max_dct_size is None else min(Hk, self.max_dct_size)
        cap_w = Wk if self.max_dct_size is None else min(Wk, self.max_dct_size)
        
       
        if self.dct_layer is None or self.dct_hw != (cap_h, cap_w):
            # scale base mappers from 7x7 to cap_h x cap_w
            mapper_x_scaled, mapper_y_scaled = self._scale_mapper(self.base_mapper_x, self.base_mapper_y, cap_h, cap_w)
            # instantiate and register new dct layer
            self.dct_layer = MultiSpectralDCTLayer(cap_h, cap_w, mapper_x_scaled, mapper_y_scaled, self.channel)
            self.dct_hw = (cap_h, cap_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]  (arbitrary H,W)
        returns: x * channel_attention (broadcasted)
        """
        B, C, H, W = x.shape
        # decide DCT grid size (we'll pool input to this size when computing DCT)
        # default: use full input size; optionally cap with max_dct_size
        dct_h = H if self.max_dct_size is None else min(H, self.max_dct_size)
        dct_w = W if self.max_dct_size is None else min(W, self.max_dct_size)

        # ensure dct_layer built for (dct_h, dct_w)
        self._ensure_dct_layer(dct_h, dct_w)

        # pool input to dct_hw (cap_h, cap_w) stored in self.dct_hw
        pool_h, pool_w = self.dct_hw
        if (H != pool_h) or (W != pool_w):
            x_pooled = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        else:
            x_pooled = x

        # compute DCT responses -> [B, C]
        y = self.dct_layer(x_pooled)  # [B, C]
        y = self.fc(y)  # [B, C]
        y = y.view(B, C, 1, 1)
        # apply gating to original x (broadcast)
        return x * y.expand_as(x)


####################################################################################
# High level FcsAttention (wraps spectral + spatial)
####################################################################################
class FcsAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16, freq_sel_method: str = 'top16',
                 max_dct_size: int = None):
        """
        in_channels: input channels (unused in original, kept for API)
        out_channels: expected channels to use for spectral layer (must equal channel dimension of x)
        reduction: FC reduction
        freq_sel_method: e.g. 'top16', 'low8'
        max_dct_size: optional cap to limit DCT filter size (memory)
        """
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be > 0")

        self.spatial = SpatialAttention()
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            reduction=reduction,
            freq_sel_method=freq_sel_method,
            max_dct_size=max_dct_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # frequency channel attention applied first
        x_freq = self.frequency_channel(x)  # x * channel_attention
        # spatial attention returns [B,1,H,W], multiply elementwise
        sp = self.spatial(x_freq)
        return x_freq * sp


