"""
Video codec utilities for encoding and decoding using DCVC models.
"""

import torch
import torch.nn.functional as F
from .image_processing import get_src_frame
from .metrics import psnr, calculate_average_ssim

try:
    from DCVC.DCVC_family.DCVC_FM.src.utils.stream_helper import get_padding_size
    from DCVC.DCVC_family.DCVC_FM.src.models.image_model import DMCI
    from DCVC.DCVC_family.DCVC_FM.src.models.video_model import DMC
except ImportError:
    print("Warning: DCVC modules not found. Video codec functionality will be limited.")
    get_padding_size = None
    DMCI = None
    DMC = None


def load_dcvc_models(model_path_i, model_path_p, device='cuda'):
    """
    Load DCVC image and video models.

    Args:
        model_path_i: path to image model checkpoint
        model_path_p: path to video model checkpoint
        device: torch device (default: 'cuda')

    Returns:
        tuple: (i_frame_net, p_frame_net)
    """
    if DMCI is None or DMC is None:
        raise ImportError("DCVC modules not available. Please install DCVC dependencies.")

    # Load image model
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(torch.load(model_path_i, map_location=device)['state_dict'])
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    # Load video model
    p_frame_net = DMC()
    p_frame_net.load_state_dict(torch.load(model_path_p, map_location=device)['state_dict'])
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()

    return i_frame_net, p_frame_net


def run_one_point_fast(p_frame_net, i_frame_net, gop, qp, fa_idx, f_num, intra_period,
                       pic_height=64, pic_width=64):
    """
    Encode a GOP (Group of Pictures) using DCVC-FM codec.

    Args:
        p_frame_net: P-frame (inter) network
        i_frame_net: I-frame (intra) network
        gop: Group of Pictures tensor [N, H, W] or [N, H, W, C]
        qp: Quantization Parameter (quality level)
        fa_idx: Frame attention index
        f_num: Number of frames
        intra_period: Intra frame period
        pic_height: Picture height (default: 64)
        pic_width: Picture width (default: 64)

    Returns:
        dict: Results containing psnrs, ssims, bits, reconstructed frames, etc.
    """
    if get_padding_size is None:
        raise ImportError("DCVC modules not available. Please install DCVC dependencies.")

    frame_num = f_num
    device = next(i_frame_net.parameters()).device

    frame_types = []
    psnrs = []
    x_hat1 = []
    ssims = []
    bits = []

    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)

    with torch.no_grad():
        for frame_idx in range(frame_num):
            x, y, u, v, rgb = get_src_frame(gop[frame_idx], device)

            # Pad if necessary
            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if frame_idx % intra_period == 0:
                # I-frame encoding
                result = i_frame_net.encode(x_padded, qp)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                # P-frame encoding
                result = p_frame_net.encode(x_padded, dpb, qp, fa_idx)
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])

            # Remove padding
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

            x_hat1.append(x_hat)

            # Calculate metrics
            curr_psnr = psnr(x_hat[0][0].cpu(), rgb[0].cpu())
            curr_ssim = calculate_average_ssim(x_hat[0][0].cpu(), rgb[0].cpu())

            psnrs.append(curr_psnr)
            ssims.append(curr_ssim)

    results = {
        'psnrs': psnrs,
        'ssims': ssims,
        'frame_types': frame_types,
        'bits': bits,
        'x_hat1': x_hat1,
    }
    return results


def calculate_total_bits(bits_list):
    """
    Calculate total bits from a list of bit values.

    Args:
        bits_list: list of bit values

    Returns:
        float: total bits
    """
    return sum(bits_list)
