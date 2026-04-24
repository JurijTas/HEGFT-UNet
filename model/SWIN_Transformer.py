#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:04:17 2026

@author: yurijtas
"""
import math
from typing import List, Tuple#, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Minimal Transformer (Swin-like) encoder (4 stages) with dynamic HW/padding
# ----------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W, f"L={L} != H*W={H*W}"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4C
        x_cat = self.norm(x_cat)
        x_reduced = self.reduction(x_cat)
        B, H2, W2, C2 = x_reduced.shape
        return x_reduced.view(B, H2 * W2, C2)

class TransformerEncoder4(nn.Module):
    """
    Produces 4 stage feature maps with channel dims:
        [embed_dim, 2*embed_dim, 4*embed_dim, 8*embed_dim]
    with automatic padding so H_pad/patch_size is divisible by 2**3.
    """
    
    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 embed_dim: int = 64,
                 depths: Tuple[int, int, int, int] = (1,1,1,1),
                 num_heads: Tuple[int, int, int, int] = (2,4,8,16)):
        
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(0.0)
        self.stage1 = nn.Sequential(*[SwinBlock(embed_dim, num_heads=num_heads[0]) for _ in range(depths[0])])
        self.merge1 = PatchMerging(embed_dim)
        self.stage2 = nn.Sequential(*[SwinBlock(embed_dim * 2, num_heads=num_heads[1]) for _ in range(depths[1])])
        self.merge2 = PatchMerging(embed_dim * 2)
        self.stage3 = nn.Sequential(*[SwinBlock(embed_dim * 4, num_heads=num_heads[2]) for _ in range(depths[2])])
        self.merge3 = PatchMerging(embed_dim * 4)
        self.stage4 = nn.Sequential(*[SwinBlock(embed_dim * 8, num_heads=num_heads[3]) for _ in range(depths[3])])
        self.n_merges = 3

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size
        n = self.n_merges
        Hp0 = math.ceil(H / p)
        Wp0 = math.ceil(W / p)
        div = 2 ** n
        Hp_target = math.ceil(Hp0 / div) * div
        Wp_target = math.ceil(Wp0 / div) * div
        H_pad = int(Hp_target * p); W_pad = int(Wp_target * p)
        pad_h = H_pad - H; pad_w = W_pad - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_pe = self.patch_embed(x)  
        x_pe = self.pos_drop(x_pe)
        Hp = x_pe.shape[2]; Wp = x_pe.shape[3]
        x_tokens = x_pe.flatten(2).transpose(1, 2)  

        x1 = self.stage1(x_tokens)
        feat1 = x1.transpose(1, 2).view(B, -1, Hp, Wp)  

        x2 = self.merge1(x1, Hp, Wp)
        Hp2, Wp2 = Hp // 2, Wp // 2
        x2 = self.stage2(x2)
        feat2 = x2.transpose(1, 2).view(B, -1, Hp2, Wp2)

        x3 = self.merge2(x2, Hp2, Wp2)
        Hp3, Wp3 = Hp2 // 2, Wp2 // 2
        x3 = self.stage3(x3)
        feat3 = x3.transpose(1, 2).view(B, -1, Hp3, Wp3)

        x4 = self.merge3(x3, Hp3, Wp3)
        Hp4, Wp4 = Hp3 // 2, Wp3 // 2
        x4 = self.stage4(x4)
        feat4 = x4.transpose(1, 2).view(B, -1, Hp4, Wp4)

        return [feat1, feat2, feat3, feat4]

