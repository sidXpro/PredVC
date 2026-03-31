"""
Metrics utilities for evaluating video quality.
"""

import math
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Warning: scikit-image not found. SSIM functionality will be limited.")
    ssim = None


def psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: first image tensor
        img2: second image tensor

    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    max_pixel = 1.0

    if mse == 0:
        return float('inf')

    return 20 * math.log10(max_pixel / math.sqrt(mse))


def calculate_average_ssim(tensor1, tensor2):
    """
    Calculate the average SSIM between two tensors of shape [N, H, W].
    Both tensors should be in [0, 1] range.

    Args:
        tensor1: first tensor [N, H, W]
        tensor2: second tensor [N, H, W]

    Returns:
        float: average SSIM value
    """
    if ssim is None:
        raise ImportError("scikit-image not available. Please install it to use SSIM.")

    # Convert tensors to numpy
    arr1 = tensor1.detach().cpu().numpy()
    arr2 = tensor2.detach().cpu().numpy()

    ssim_list = []
    for i in range(arr1.shape[0]):
        ssim_val = ssim(arr1[i], arr2[i], data_range=1.0)
        ssim_list.append(ssim_val)

    return np.mean(ssim_list)


def calculate_bpp(total_bits, height, width, num_frames):
    """
    Calculate bits per pixel (BPP).

    Args:
        total_bits: total number of bits
        height: frame height
        width: frame width
        num_frames: number of frames

    Returns:
        float: bits per pixel
    """
    total_pixels = height * width * num_frames
    return total_bits / total_pixels
