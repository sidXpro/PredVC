"""
Image processing utilities for video prediction and compression.
"""

import torch
import numpy as np

try:
    from DCVC.DCVC_family.DCVC_FM.src.transforms.functional import rgb_to_ycbcr420, rgb_to_ycbcr444
except ImportError:
    print("Warning: DCVC modules not found. Some functionality may be limited.")
    rgb_to_ycbcr444 = None


def np_image_to_tensor(img):
    """
    Convert a numpy image to a PyTorch tensor.

    Args:
        img: numpy array of shape (H, W) or (H, W, C)

    Returns:
        torch.FloatTensor with added batch dimension
    """
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def get_src_frame(rgb, device):
    """
    Prepare a source frame for encoding by converting to YCbCr444 format.

    Args:
        rgb: RGB image as numpy array
        device: torch device (cpu or cuda)

    Returns:
        tuple: (x, y, u, v, rgb) where x is the tensor ready for encoding
    """
    if rgb_to_ycbcr444 is None:
        raise ImportError("DCVC modules not available. Please install DCVC dependencies.")

    y, uv = rgb_to_ycbcr444(rgb)

    u, v = None, None
    yuv = np.concatenate((y, uv), axis=0)
    x = np_image_to_tensor(yuv)

    x = x.to(device)
    return x, y, u, v, rgb


def normalize_max_min(tensor1):
    """
    Normalize tensor to [0, 1] range using min-max normalization.

    Args:
        tensor1: input tensor

    Returns:
        normalized tensor
    """
    return (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min())


def adjust_range(tensor1, original_tensor):
    """
    Adjust tensor range to match the range of an original tensor.

    Args:
        tensor1: tensor to adjust
        original_tensor: reference tensor with target range

    Returns:
        adjusted tensor
    """
    tensor2 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min())
    tensor3 = tensor2 * (original_tensor.max() - original_tensor.min()) + original_tensor.min()
    return tensor3


def adjust_stats(tensor1, original_tensor):
    """
    Adjust tensor to match the mean and standard deviation of an original tensor.

    Args:
        tensor1: tensor to adjust
        original_tensor: reference tensor with target statistics

    Returns:
        adjusted tensor
    """
    return ((tensor1 - tensor1.mean()) / tensor1.std()) * original_tensor.std() + original_tensor.mean()
