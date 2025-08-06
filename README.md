# Mars Super Resolution

A CNN-based approach for memory-efficient super-resolution on high-resolution images. This project placed 1st in the NEXUS (Inter-Branch ML Forge) Competition.

## Overview

MARS Super Resolution delivers robust 2×/4× upscaling for images without leveraging any pre-trained weights. Designed for research and competition constraints, it uses an original convolutional neural network architecture engineered from scratch.

## Key Features

- **Patch-based Training:**  
  Utilizes the DIV2K dataset with a custom patch-based pipeline — dramatically reducing memory consumption while allowing for high-resolution detail learning.

- **PixelShuffle Upsampling:**  
  Implements sub-pixel convolution (PixelShuffle) layers for precise spatial upsampling, minimizing artifacts and maintaining texture fidelity.

- **Overlapping Patch Stitching:**  
  Seamless reconstruction of full-resolution images using overlapping windowing techniques, eliminating blockiness and edge artifacts.

- **No Pre-trained Models:**  
  The model is trained entirely from scratch, ensuring the results showcase the real capability of the architecture and data handling, with no borrowed weights.

- **Restoration Quality:**  
  Achieves state-of-the-art results for PSNR and SSIM metrics compared to popular baseline models, outperforming other solutions under strict “no pre-trained models” rules.

## Highlights

- Original architecture tailored for both efficiency and quality.
- Pure PyTorch implementation, modular and easy to extend.
- Suitable for research experiments or as a reproducible competition baseline.

## Usage

1. **Model:**  
   Find the trained model in "model"  folder.

2. **Inference:**  
   Use `infer.py` for super-resolving input images with 2× or 4× scaling.

