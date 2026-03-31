#!/usr/bin/env python3
"""
PredVC - Video Prediction and Compression Main CLI

A comprehensive tool for video frame prediction using diffusion models (MCVD)
and compression using DCVC codecs.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.diffusion_models import load_diffusion_model, get_ddim_sampler, autoregressive_predict_v2
from utils.video_codec import load_dcvc_models, run_one_point_fast, calculate_total_bits
from utils.metrics import psnr, calculate_average_ssim, calculate_bpp
from utils.image_processing import normalize_max_min


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PredVC - Video Prediction and Compression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['predict', 'compress', 'full_pipeline', 'benchmark'],
                        help='Task to perform: predict (frame prediction only), '
                             'compress (compression only), full_pipeline (predict + compress), '
                             'or benchmark (evaluation)')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kth', 'bair', 'cityscapes'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')

    # Model paths
    parser.add_argument('--diffusion-ckpt', type=str,
                        help='Path to diffusion model checkpoint')
    parser.add_argument('--dcvc-image-model', type=str,
                        help='Path to DCVC image model checkpoint')
    parser.add_argument('--dcvc-video-model', type=str,
                        help='Path to DCVC video model checkpoint')

    # Prediction parameters
    parser.add_argument('--num-pred-frames', type=int, default=25,
                        help='Number of frames to predict')
    parser.add_argument('--num-cond-frames', type=int, default=10,
                        help='Number of conditional frames')
    parser.add_argument('--pred-len', type=int, default=5,
                        help='Frames predicted per step')
    parser.add_argument('--use-pred-for-cond', type=int, default=2,
                        help='Number of predicted frames used for next condition')
    parser.add_argument('--subsample', type=int, default=100,
                        help='DDIM sampling steps')

    # Compression parameters
    parser.add_argument('--qp', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 40, 50, 60, 63],
                        help='Quantization parameters for compression')
    parser.add_argument('--intra-period', type=int, default=6,
                        help='Intra frame period')
    parser.add_argument('--fa-idx', type=int, default=0,
                        help='Frame attention index')
    parser.add_argument('--frame-height', type=int, default=64,
                        help='Frame height')
    parser.add_argument('--frame-width', type=int, default=64,
                        help='Frame width')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save reconstructed frames')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')

    # Batch processing
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None = all)')

    return parser.parse_args()


def run_prediction(args):
    """Run frame prediction task."""
    print(f"\n{'='*60}")
    print("Running Frame Prediction")
    print(f"{'='*60}\n")

    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load diffusion model
    print(f"Loading diffusion model from {args.diffusion_ckpt}...")
    scorenet, config = load_diffusion_model(args.diffusion_ckpt, device)

    # Update config with arguments
    config.data.num_frames = args.num_pred_frames
    config.data.num_frames_cond = args.num_cond_frames
    config.training.batch_size = args.batch_size

    # Get sampler
    print(f"Initializing DDIM sampler with {args.subsample} steps...")
    sampler = get_ddim_sampler(config, scorenet, subsample=args.subsample, verbose=args.verbose)

    # Load dataset
    print(f"Loading {args.dataset} dataset from {args.data_path}...")
    from datasets import get_dataset
    dataset, test_dataset = get_dataset(args.data_path, config,
                                       video_frames_pred=config.data.num_frames)

    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                            shuffle=False, num_workers=0, drop_last=True)

    # Process samples
    results = []
    num_processed = 0
    max_samples = args.num_samples if args.num_samples else len(test_loader)

    print(f"\nPredicting frames for {max_samples} samples...")

    for i, batch in enumerate(test_loader):
        if num_processed >= max_samples:
            break

        # Get conditional and real frames
        test_x, real_x = batch
        cond = test_x[:, :config.data.num_frames_cond].to(device)
        real = real_x.to(device)

        # Create conditioning mask
        cond_mask = torch.zeros(1, config.data.num_frames_cond + config.data.num_frames,
                               device=device)
        cond_mask[:, :config.data.num_frames_cond] = 1.0

        print(f"\nSample {i+1}/{max_samples}")
        print(f"  Conditional frames: {cond.shape}")
        print(f"  Real frames: {real.shape}")

        # Run prediction
        with torch.no_grad():
            predicted_frames = autoregressive_predict_v2(
                sampler=sampler,
                scorenet=scorenet,
                cond=cond,
                cond_mask=cond_mask,
                num_pred_frames=args.num_pred_frames,
                cond_len=args.num_cond_frames,
                pred_len=args.pred_len,
                use_pred_for_cond=args.use_pred_for_cond,
                subsample=args.subsample,
                verbose=args.verbose,
                device=device
            )

        # Calculate metrics if real frames available
        if predicted_frames.shape[0] <= real.shape[1]:
            real_for_metric = real[0, :predicted_frames.shape[0]].cpu()
            pred_for_metric = predicted_frames.cpu()

            avg_psnr = np.mean([psnr(pred_for_metric[j:j+1], real_for_metric[j:j+1])
                               for j in range(predicted_frames.shape[0])])
            avg_ssim = calculate_average_ssim(pred_for_metric, real_for_metric)

            print(f"  Avg PSNR: {avg_psnr:.2f} dB")
            print(f"  Avg SSIM: {avg_ssim:.4f}")

            results.append({
                'sample_idx': i,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim
            })

        # Save frames if requested
        if args.save_frames:
            output_dir = Path(args.output_dir) / 'predicted_frames' / f'sample_{i:04d}'
            output_dir.mkdir(parents=True, exist_ok=True)

            for j, frame in enumerate(predicted_frames):
                frame_path = output_dir / f'frame_{j:04d}.npy'
                np.save(frame_path, frame.cpu().numpy())

            print(f"  Saved frames to {output_dir}")

        num_processed += 1

    # Save results
    results_path = Path(args.output_dir) / 'prediction_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Prediction complete! Results saved to {results_path}")
    print(f"{'='*60}\n")

    return results


def run_compression(args):
    """Run compression task."""
    print(f"\n{'='*60}")
    print("Running Video Compression")
    print(f"{'='*60}\n")

    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load DCVC models
    print("Loading DCVC models...")
    i_frame_net, p_frame_net = load_dcvc_models(
        args.dcvc_image_model,
        args.dcvc_video_model,
        device
    )

    # TODO: Load video data
    # This would need to be implemented based on the specific dataset format
    print("Note: Compression requires video data to be loaded.")
    print("Please implement data loading based on your specific dataset format.")

    return {}


def run_full_pipeline(args):
    """Run full prediction + compression pipeline."""
    print(f"\n{'='*60}")
    print("Running Full Pipeline (Prediction + Compression)")
    print(f"{'='*60}\n")

    # First run prediction
    print("\n--- Step 1: Frame Prediction ---")
    prediction_results = run_prediction(args)

    # Then run compression on predicted frames
    print("\n--- Step 2: Compression ---")
    # TODO: Implement compression on predicted frames
    print("Note: Compression of predicted frames to be implemented")

    return prediction_results


def run_benchmark(args):
    """Run benchmark evaluation."""
    print(f"\n{'='*60}")
    print("Running Benchmark Evaluation")
    print(f"{'='*60}\n")

    # TODO: Implement benchmarking across multiple QP values
    print("Note: Benchmarking to be implemented")

    return {}


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'='*60}")
    print("PredVC - Video Prediction and Compression")
    print(f"{'='*60}")
    print(f"\nTask: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run selected task
    if args.task == 'predict':
        results = run_prediction(args)
    elif args.task == 'compress':
        results = run_compression(args)
    elif args.task == 'full_pipeline':
        results = run_full_pipeline(args)
    elif args.task == 'benchmark':
        results = run_benchmark(args)
    else:
        print(f"Unknown task: {args.task}")
        sys.exit(1)

    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
