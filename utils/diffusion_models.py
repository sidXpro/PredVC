"""
Diffusion model utilities for video frame prediction using MCVD.
"""

import torch

try:
    from load_model_from_ckpt import load_model, get_sampler, init_samples
    from datasets import get_dataset, data_transform, inverse_data_transform
    from runners.ncsn_runner import conditioning_fn
except ImportError:
    print("Warning: MCVD modules not found. Diffusion model functionality will be limited.")
    load_model = None
    get_sampler = None


def load_diffusion_model(ckpt_path, device='cuda'):
    """
    Load MCVD diffusion model from checkpoint.

    Args:
        ckpt_path: path to model checkpoint
        device: torch device (default: 'cuda')

    Returns:
        tuple: (scorenet, config)
    """
    if load_model is None:
        raise ImportError("MCVD modules not available. Please ensure MCVD is properly installed.")

    scorenet, config = load_model(ckpt_path, device)
    scorenet.eval()

    return scorenet, config


def get_ddim_sampler(config, scorenet, subsample=100, verbose=False):
    """
    Get DDIM sampler for frame prediction.

    Args:
        config: model configuration
        scorenet: loaded score network
        subsample: number of DDIM steps (default: 100)
        verbose: verbose output (default: False)

    Returns:
        sampler function
    """
    if get_sampler is None:
        raise ImportError("MCVD modules not available.")

    sampler = get_sampler(config, scorenet, subsample=subsample, verbose=verbose)
    return sampler


def autoregressive_predict_v2(
    sampler,
    scorenet,
    cond,
    cond_mask,
    num_pred_frames=25,
    cond_len=10,
    pred_len=5,
    use_pred_for_cond=2,
    subsample=100,
    verbose=True,
    init=None,
    device='cuda'
):
    """
    Predicts video frames autoregressively using diffusion models.

    Each prediction step generates pred_len frames, but only use_pred_for_cond frames
    are used for the next step's conditioning (combined with the last frames from
    the previous condition).

    Args:
        sampler: DDIM sampler function
        scorenet: score network model
        cond: Initial condition tensor [1, cond_len, H, W]
        cond_mask: Conditioning mask
        num_pred_frames: Total number of frames to predict (default: 25)
        cond_len: Length of conditioning sequence (default: 10)
        pred_len: Number of frames predicted per step (default: 5)
        use_pred_for_cond: Number of predicted frames to use in next condition (default: 2)
        subsample: DDIM sampling steps (default: 100)
        verbose: Print progress information (default: True)
        init: Initial noise (default: None)
        device: torch device (default: 'cuda')

    Returns:
        torch.Tensor: Predicted frames [num_pred_frames, H, W]
    """
    cond_frames = cond.clone()  # [1, cond_len, H, W]
    predicted_frames = []

    total_generated = 0
    while total_generated < num_pred_frames:
        # Generate frames using the sampler
        pred = sampler(
            init=init,
            scorenet=scorenet,
            cond=cond_frames,      # [1, cond_len, H, W]
            cond_mask=cond_mask,
            subsample=subsample,
            verbose=verbose
        )  # [1, pred_len, H, W]

        # How many frames left to predict?
        frames_to_append = min(pred_len, num_pred_frames - total_generated)

        # Keep only the frames we need
        predicted_frames.append(pred[:, :frames_to_append])

        total_generated += frames_to_append

        # Prepare conditioning for next step
        # Use last (cond_len - use_pred_for_cond) frames from current condition
        # Plus first use_pred_for_cond frames from prediction
        last_cond = cond_frames[:, -(cond_len - use_pred_for_cond):, :, :]  # [1, 8, H, W]
        last_cond = last_cond.to(device)

        pred_for_cond = pred[:, :use_pred_for_cond, :, :]  # [1, 2, H, W]
        pred_for_cond = pred_for_cond.to(device)

        cond_frames = torch.cat([last_cond, pred_for_cond], dim=1)  # [1, 10, H, W]

    # Concatenate all predicted frames
    predicted_frames_tensor = torch.cat(predicted_frames, dim=1)  # [1, num_pred_frames, H, W]
    predicted_frames_tensor = predicted_frames_tensor.squeeze(0)   # [num_pred_frames, H, W]

    return predicted_frames_tensor
