#!/bin/bash
# Example usage scripts for PredVC

# Example 1: Frame Prediction on KTH Dataset
# Replace paths with your actual model and data paths
echo "Example 1: Frame Prediction"
echo "python main.py \\"
echo "  --task predict \\"
echo "  --dataset kth \\"
echo "  --data-path /path/to/KTH64_h5 \\"
echo "  --diffusion-ckpt /path/to/checkpoint_400000.pt \\"
echo "  --num-pred-frames 25 \\"
echo "  --num-cond-frames 10 \\"
echo "  --subsample 100 \\"
echo "  --output-dir ./output/kth_prediction \\"
echo "  --save-frames \\"
echo "  --device cuda"
echo ""

# Example 2: Video Compression
echo "Example 2: Video Compression"
echo "python main.py \\"
echo "  --task compress \\"
echo "  --dataset kth \\"
echo "  --data-path /path/to/video_data \\"
echo "  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \\"
echo "  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \\"
echo "  --qp 10 20 30 40 \\"
echo "  --intra-period 6 \\"
echo "  --frame-height 64 \\"
echo "  --frame-width 64 \\"
echo "  --output-dir ./output/compression \\"
echo "  --device cuda"
echo ""

# Example 3: Full Pipeline (BAIR Dataset)
echo "Example 3: Full Pipeline"
echo "python main.py \\"
echo "  --task full_pipeline \\"
echo "  --dataset bair \\"
echo "  --data-path /path/to/BAIR_h5 \\"
echo "  --diffusion-ckpt /path/to/bair_checkpoint.pt \\"
echo "  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \\"
echo "  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \\"
echo "  --num-pred-frames 25 \\"
echo "  --qp 10 20 30 \\"
echo "  --output-dir ./output/bair_full \\"
echo "  --device cuda"
echo ""

# Example 4: Benchmark
echo "Example 4: Benchmark Evaluation"
echo "python main.py \\"
echo "  --task benchmark \\"
echo "  --dataset kth \\"
echo "  --data-path /path/to/KTH64_h5 \\"
echo "  --diffusion-ckpt /path/to/checkpoint_400000.pt \\"
echo "  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \\"
echo "  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \\"
echo "  --qp 5 10 15 20 25 30 40 50 60 63 \\"
echo "  --output-dir ./output/benchmark \\"
echo "  --device cuda"
echo ""

# Show help
echo "For full list of options, run:"
echo "python main.py --help"
