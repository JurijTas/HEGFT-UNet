

from typing import  Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_smp_encoder import CNNEncoder
from .MFFB import MFFB
from .EFFB  import EFFB
from .SWIN_Transformer import TransformerEncoder4
from .decoder import UNetMFCSADecoder

#------------------------------------------
#
#                 HEGFTUNeT
#
#-----------------------------------------  
    
class BottleNeckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeckLayer, self).__init__()
        # support case where in_channels != out_channels (e.g. 512 -> 1024)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def initialize_weights(module):
    """Custom weight initialization."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    
    
    
    
class HEGFTUNeT(nn.Module):
    def __init__(self,
                 cnn_encoder_name: str = "resnet34",
                 cnn_encoder_weights: str = None,
                 cnn_in_channels: int = 1,
                 img_size: int = 256,
                 patch_size: int = 4,
                 embed_dim: int = 64,
                 transformer_depths: tuple = (1,1,1,1),
                 transformer_heads: tuple = (2,4,8,16),
                 cnn_channels: Optional[List[int]] = None,
                 trans_channels: Optional[List[int]] = None,
                 num_classes: int = 1,
                 freq_sel_method: str="top16",
                 use_jt_fusion_block: bool = True):
        super().__init__()
        
        
        self.config = {
            'cnn_encoder_name': cnn_encoder_name,
            'cnn_encoder_weights': cnn_encoder_weights,
            'img_size': img_size,
            'patch_size': patch_size,
            'embed_dim': embed_dim,
            'transformer_depths' : transformer_depths,
            'transformer_heads' : transformer_heads,
            'cnn_channels' : cnn_channels,
            'trans_channels' :  trans_channels,
            'num_classes': num_classes,
            'use_jt_fusion_block': use_jt_fusion_block,
            'freq_sel_method': freq_sel_method
            }
        

        # CNN encoder
        
        self.cnn_encoder = CNNEncoder(encoder_name=cnn_encoder_name,
                                      encoder_weights=cnn_encoder_weights,
                                      input_channels=cnn_in_channels)

        if cnn_channels is None:
            out_chs = self.cnn_encoder.out_channels
            if len(out_chs) >= 4:
                cnn_channels = out_chs[-4:]
            else:
                raise ValueError("CNN encoder out_channels length < 4; pass cnn_channels explicitly.")
                
        self.cnn_channels = cnn_channels
        self.cnn_in_channels = cnn_in_channels
        self.freq_sel_method = freq_sel_method

        # Transformer encoder
        self.tr_encoder = TransformerEncoder4(img_size=img_size,
                                              patch_size=patch_size,
                                              in_chans=cnn_in_channels,
                                              embed_dim=embed_dim,
                                              depths=transformer_depths,
                                              num_heads=transformer_heads)

        if trans_channels is None:
            trans_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        self.trans_channels = trans_channels

        # Build fusion modules: use MFFB for each stage when requested.
        self.use_jt_fusion_block = use_jt_fusion_block
        self.fusion_modules = nn.ModuleList()
        for i in range(4):
            
            out_ch = self.cnn_channels[i]

            if use_jt_fusion_block:
                
                self.fusion_modules.append(MFFB(self.trans_channels[i],
                                                           self.cnn_channels[i],
                                                           out_ch))
            else:
                
                in_ch = self.trans_channels[i] + self.cnn_channels[i]
                self.fusion_modules.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
    
        #  Bundary Enchancement Module
        self.EFFB_module =  EFFB(self.cnn_channels[-1],self.trans_channels[-1]) 
                
        # BottleNeck   
        
        self.bottleneck = BottleNeckLayer(in_channels=2*self.cnn_channels[-1], out_channels=2*self.cnn_channels[-1])
        # Decoder
        self.decoder = UNetMFCSADecoder(skip_channels=self.cnn_channels,
                                            bottleneck_ch=2*self.cnn_channels[-1],
                                            num_classes=num_classes,
                                            freq_sel_method = self.freq_sel_method)
        
        
      

        # when last Transformer layer and cnn layer have different number of channels
        # we use this module to create bottleneck(1024) input
        self.reduce_conv_last = nn.Sequential(
          nn.Conv2d(self.cnn_channels[-1] + self.trans_channels[-1],  2*self.cnn_channels[-1], kernel_size=1, bias=False),
          nn.BatchNorm2d(2*self.cnn_channels[-1]),
          nn.ReLU(inplace=True)
         )
        # ---- Apply initialization only to new (non-pretrained) parts
        if cnn_encoder_weights is None:
           self.cnn_encoder.apply(initialize_weights)
        self.tr_encoder.apply(initialize_weights)
        self.fusion_modules.apply(initialize_weights)
        self.bottleneck.apply(initialize_weights)
        self.decoder.apply(initialize_weights)
        self.reduce_conv_last.apply(initialize_weights)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[2], x.shape[3]
        # get cnn features (list) and transformer features (list)
        cnn_feats_all = self.cnn_encoder(x)
        if len(cnn_feats_all) >= 4:
            cnn_feats = cnn_feats_all[-4:]
        else:
            raise ValueError("CNN encoder returned fewer than 4 feature maps")

        tr_feats = self.tr_encoder(x)
        if len(tr_feats) != 4:
            raise ValueError("Transformer encoder must return 4 stages")

        fused_skips = []
        for i, (cfeat, tfeat, fus_m) in enumerate(zip(cnn_feats, tr_feats, self.fusion_modules)):

            # align transformer spatial size to CNN skip if needed
            if tfeat.shape[2:] != cfeat.shape[2:]:
                t_aligned = F.interpolate(tfeat, size=cfeat.shape[2:], mode='bilinear', align_corners=False)
            else:
                t_aligned = tfeat

            is_last = (i == len(self.fusion_modules) - 1)

            if self.use_jt_fusion_block:
                if is_last:
                   fused = fus_m(t_aligned, cfeat) + self.EFFB_module(cfeat, t_aligned) ### EFFB
                else:
                   fused = fus_m(t_aligned, cfeat) 
              
            else:
                if is_last:
                   fused = fus_m(torch.cat([t_aligned, cfeat], dim=1)) +  self.EFFB_module(cfeat, t_aligned)## No BEM
                else:
                   fused = fus_m(torch.cat([t_aligned, cfeat], dim=1))
            
            if is_last:
                bottleneck_feat = torch.cat([t_aligned, cfeat], dim=1)
           
            fused_skips.append(fused)   
            
        if bottleneck_feat.shape[1] != 2 * self.cnn_channels[-1]:
             bottleneck_feat = self.reduce_conv_last(bottleneck_feat)

        bottleneck = self.bottleneck(bottleneck_feat)  

        out = self.decoder(bottleneck, fused_skips,out_size=orig_hw)
        
        return out


# # ----------------------------
# Quick runtime test (grayscale input)
# ----------------------------
if __name__ == "__main__":
    #
    model = HEGFTUNeT(cnn_encoder_name="resnet34", cnn_encoder_weights='imagenet',
                             cnn_in_channels=1, patch_size=4, embed_dim=64, num_classes=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(2, 1, 256, 256).to(device)   # batch of grayscale images (H != W)
    with torch.no_grad():
        y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
### and that's it !!!  ###

