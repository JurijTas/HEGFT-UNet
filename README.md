# HEGFT-UNet: Hybrid Edge-Guided Frequency-Aware Transformer for Thyroid Nodule Segmentation
- This repository contains the official implementation of HEGFT-UNet, a deep learning framework designed for accurate thyroid nodule segmentation in ultrasound images.
- HEGFT-UNet addresses the inherent challenges of medical ultrasound, such as low contrast, speckle noise, and hardware variability, by integrating global context modeling with frequency-aware feature refinement.
## Key Features:
- Hybrid Dual-Encoder: Combines CNN-based local feature extraction with Swin Transformer (ST) to jointly capture fine-grained textures and long-range global dependencies.
- MFCSA Decoder: A novel Multi-Frequency Channel Spatial Attention module that utilizes Discrete Cosine Transform (DCT) cues to suppress noise and emphasize diagnostically relevant anatomical details.
- Edge-Aware Guidance: Implements a multi-scale feature aggregation strategy focused on structural consistency and precise boundary preservation, especially for small lesions.
- Robust Generalization: Proven performance across diverse clinical scenarios through cross-dataset validation (trained on TN3K and validated on the independent IPPT clinical cohort).
## Architecture Overview:
The HEGFT-UNet framework is built upon three primary contributions:
- Global-Local Integration: A dual-stream architecture ensuring effective representation of both macro-anatomical structures and micro-textures.
- Boundary-Aware Refinement: A multi-stage fusion strategy that strengthens edge consistency across nodules of varying sizes.
- Frequency-Domain Learning: Integration of frequency cues to enhance robustness against the variability of ultrasound protocols and hardware environments.
## Datasets:
To ensure clinical relevance and reproducibility, the model was evaluated using:
- TN3K Dataset: A large-scale public dataset for thyroid ultrasound segmentation.
- IPPT Dataset: An independent clinical cohort used to demonstrate the framework's robust generalization in real-world environments.
