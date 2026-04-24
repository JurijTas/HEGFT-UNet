

from typing import Optional
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch


class CNNEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = None,
        input_channels: int = 1
    ):
        super().__init__()

        if smp is None:
            raise ImportError("segmentation_models_pytorch (smp) is required for CNNEncoder.")

        # ------------------------------------------------------------
        # CASE 1 — Using pretrained weights
        # ------------------------------------------------------------
        if encoder_weights is not None:
            print(f"Loading pretrained encoder (smp) '{encoder_name}' with weights='{encoder_weights}'")

            # Must load with 3 channels to receive pretrained weights
            self.encoder = smp.encoders.get_encoder(
                encoder_name,
                in_channels=3,
                weights=encoder_weights
            )

            # ------------------------------------------------------------
            # If input_channels != 3 → adapt conv1 using safe copy
            # ------------------------------------------------------------
            if input_channels != 3:
                print(f"  Adapting pretrained conv1 from 3 → {input_channels} channels")

                old_conv = self.encoder.conv1
                w = old_conv.weight    # (out_ch, 3, k, k)

                # New conv configured for desired # of channels
                new_conv = nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None)
                )

                # SAFE weight adaptation using torch.no_grad()
                with torch.no_grad():
                    # Average RGB → grayscale → repeat for extra channels
                    w_avg = w.mean(dim=1, keepdim=True)  # [out_ch,1,k,k]
                    new_w = w_avg.repeat(1, input_channels, 1, 1)

                    new_conv.weight.copy_(new_w)

                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)

                # Replace the original conv1
                self.encoder.conv1 = new_conv

        # ------------------------------------------------------------
        # CASE 2 — No pretrained weights (fresh init)
        # ------------------------------------------------------------
        else:
            print(f"Initializing encoder '{encoder_name}' from scratch (no pretrained weights)")
            self.encoder = smp.encoders.get_encoder(
                encoder_name,
                in_channels=input_channels,
                weights=None
            )

        # Required for decoder building
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        return self.encoder(x)



# ----------------------------
# CNN Encoder wrapper (SMP) supporting variable input channels
# ----------------------------
# class CNNEncoder(nn.Module):
#     def __init__(self, encoder_name: str = "resnet34", encoder_weights: Optional[str] = None, input_channels: int = 1):
#         super().__init__()
#         if smp is None:
#             raise ImportError("smp is required for CNNEncoder. Install segmentation-models-pytorch or use the Dummy fallback.")
#         # If using pretrained weights with non-3-channel input, don't request pretrained weights (incompatible)
#         use_weights = encoder_weights
#         if input_channels != 3 and encoder_weights is not None:
#             print("Warning: pretrained imagenet weights expect 3 input channels; disabling weights to avoid mismatch.")
#             use_weights = None
#         self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=input_channels, weights=use_weights)
#         self.out_channels = self.encoder.out_channels

#     def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
#         return self.encoder(x)




