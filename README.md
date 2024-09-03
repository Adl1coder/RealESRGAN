# RealESRGAN
Real-ESRGAN (Real-Enhanced Super-Resolution Generative Adversarial Networks) is a state-of-the-art method for enhancing the resolution of images using deep learning techniques
# Real-ESRGAN Image Enhancement

This project uses Real-ESRGAN to enhance the resolution of images using deep learning techniques. It leverages Generative Adversarial Networks (GANs) to upscale images while preserving high-quality details and textures.

## Installation

Ensure you have the required packages installed. You can install them using pip:

```bash
pip install basicsr
pip install facexlib
pip install gfpgan
pip install realesrgan
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGAN
from google.colab import files
import os
from math import ceil, sqrt
import numpy as np
from PIL import Image

def enhance_image(input_file, layers=2, upscale=2, final_filename="", enhance_faces=False):
    if layers == 4:
        # 4 layers
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_file = 'realesr-general-x4v3.pth'  # Ensure this file is in the same directory as your script
    elif layers == 2:
        # 2 layers
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        model_file = 'realesr-general-x2v3.pth'  # Ensure this file is in the same directory as your script

    # Load model and enhance image
    # Add your image enhancement logic here

