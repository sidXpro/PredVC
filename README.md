# PredVC

**Video Prediction and Compression using Diffusion Models and Neural Codecs**

PredVC is a comprehensive framework for video frame prediction using MCVD (Masked Conditional Video Diffusion) models and video compression using DCVC (Deep Contextual Video Compression) neural codecs.

## Features

- **Frame Prediction**: Use diffusion models (MCVD) to predict future video frames
- **Video Compression**: Compress videos using state-of-the-art neural codecs (DCVC-DC, DCVC-FM)
- **Full Pipeline**: Combine prediction and compression for efficient video encoding
- **Benchmarking**: Evaluate performance across multiple datasets and quality levels
- **Modular Design**: Well-organized utilities for easy extension and modification

## Repository Structure

```
PredVC/
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── image_processing.py     # Image preprocessing and transformations
│   ├── metrics.py              # PSNR, SSIM, and other quality metrics
│   ├── video_codec.py          # DCVC codec wrapper functions
│   └── diffusion_models.py     # MCVD diffusion model utilities
├── main.py                     # Main CLI application
├── requirements.txt            # Python dependencies
├── KTH2_MCVD_DCVC1.ipynb      # Original KTH dataset notebook
├── BAIR_MCVD_DCVC1.ipynb      # Original BAIR dataset notebook
├── Graphs_Benchmarks_VC.ipynb # Original benchmarking notebook
└── README.md                   # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sidXpro/PredVC.git
cd PredVC
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install DCVC (Deep Contextual Video Compression)

```bash
# Clone DCVC repository
git clone https://github.com/microsoft/DCVC
cd DCVC

# Install DCVC-DC (CVPR 2023)
cd DCVC_family/DCVC_DC
pip install -r requirements.txt

# Install DCVC-FM (CVPR 2024)
cd ../DCVC_FM
pip install -r requirements.txt

cd ../../..
```

### 4. Install MCVD (Masked Conditional Video Diffusion)

```bash
# Clone MCVD repository
git clone https://github.com/voletiv/mcvd-pytorch
cd mcvd-pytorch
pip install -r requirements.txt
cd ..
```

### 5. Download Pre-trained Models

Download the required model checkpoints:

**DCVC Models:**
- DCVC-DC: Download from [DCVC releases](https://github.com/microsoft/DCVC/releases)
- DCVC-FM: Download from [DCVC releases](https://github.com/microsoft/DCVC/releases)

**MCVD Models:**
- Download from the MCVD repository or train your own

## Usage

### Command Line Interface

The main CLI provides four primary tasks:

#### 1. Frame Prediction Only

Predict future video frames using diffusion models:

```bash
python main.py \
  --task predict \
  --dataset kth \
  --data-path /path/to/KTH_dataset \
  --diffusion-ckpt /path/to/mcvd_checkpoint.pt \
  --num-pred-frames 25 \
  --num-cond-frames 10 \
  --subsample 100 \
  --output-dir ./output/prediction \
  --save-frames \
  --device cuda
```

#### 2. Video Compression Only

Compress video frames using DCVC codec:

```bash
python main.py \
  --task compress \
  --dataset kth \
  --data-path /path/to/video_data \
  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \
  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \
  --qp 10 20 30 40 \
  --intra-period 6 \
  --frame-height 64 \
  --frame-width 64 \
  --output-dir ./output/compression \
  --device cuda
```

#### 3. Full Pipeline (Prediction + Compression)

Run the complete pipeline:

```bash
python main.py \
  --task full_pipeline \
  --dataset kth \
  --data-path /path/to/KTH_dataset \
  --diffusion-ckpt /path/to/mcvd_checkpoint.pt \
  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \
  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \
  --num-pred-frames 25 \
  --qp 10 20 30 \
  --output-dir ./output/full_pipeline \
  --device cuda
```

#### 4. Benchmark Evaluation

Evaluate across multiple quality levels:

```bash
python main.py \
  --task benchmark \
  --dataset kth \
  --data-path /path/to/KTH_dataset \
  --diffusion-ckpt /path/to/mcvd_checkpoint.pt \
  --dcvc-image-model /path/to/cvpr2024_image.pth.tar \
  --dcvc-video-model /path/to/cvpr2024_video.pth.tar \
  --qp 5 10 15 20 25 30 40 50 60 63 \
  --output-dir ./output/benchmark \
  --device cuda
```

### Important Parameters

#### Dataset Parameters
- `--dataset`: Dataset to use (`kth`, `bair`, `cityscapes`)
- `--data-path`: Path to the dataset directory

#### Model Paths
- `--diffusion-ckpt`: Path to MCVD diffusion model checkpoint
- `--dcvc-image-model`: Path to DCVC image (I-frame) model
- `--dcvc-video-model`: Path to DCVC video (P-frame) model

#### Prediction Parameters
- `--num-pred-frames`: Number of frames to predict (default: 25)
- `--num-cond-frames`: Number of conditional frames (default: 10)
- `--pred-len`: Frames predicted per step (default: 5)
- `--use-pred-for-cond`: Predicted frames used for next condition (default: 2)
- `--subsample`: DDIM sampling steps (default: 100)

#### Compression Parameters
- `--qp`: Quantization parameters, higher = more compression (default: [5, 10, 15, 20, 25, 30, 40, 50, 60, 63])
- `--intra-period`: I-frame interval (default: 6)
- `--fa-idx`: Frame attention index (default: 0)
- `--frame-height`: Frame height (default: 64)
- `--frame-width`: Frame width (default: 64)

#### Output Parameters
- `--output-dir`: Directory for saving results (default: ./output)
- `--save-frames`: Save reconstructed frames
- `--verbose`: Enable verbose output

#### System Parameters
- `--device`: Computation device (`cuda` or `cpu`, default: cuda)
- `--batch-size`: Batch size for processing (default: 1)
- `--num-samples`: Number of samples to process (default: all)

## Example Workflows

### KTH Dataset Processing

```bash
# Predict frames for KTH dataset
python main.py \
  --task predict \
  --dataset kth \
  --data-path /data/KTH64_h5 \
  --diffusion-ckpt /models/kth64_checkpoint_400000.pt \
  --num-pred-frames 25 \
  --num-cond-frames 10 \
  --output-dir ./output/kth_prediction \
  --save-frames \
  --device cuda
```

### BAIR Dataset Processing

```bash
# Full pipeline for BAIR dataset
python main.py \
  --task full_pipeline \
  --dataset bair \
  --data-path /data/BAIR_h5 \
  --diffusion-ckpt /models/bair64_checkpoint.pt \
  --dcvc-image-model /models/cvpr2024_image.pth.tar \
  --dcvc-video-model /models/cvpr2024_video.pth.tar \
  --output-dir ./output/bair_full \
  --device cuda
```

## Using the Python Modules Directly

You can also import and use the utilities in your own Python scripts:

```python
import torch
from utils.diffusion_models import load_diffusion_model, autoregressive_predict_v2
from utils.video_codec import load_dcvc_models, run_one_point_fast
from utils.metrics import psnr, calculate_average_ssim

# Load models
device = 'cuda'
scorenet, config = load_diffusion_model('checkpoint.pt', device)
i_net, p_net = load_dcvc_models('image_model.pt', 'video_model.pt', device)

# Your custom processing code here...
```

## Performance Tips

1. **GPU Memory**: For large frame sizes or long sequences, reduce batch size or use CPU offloading
2. **Sampling Steps**: Fewer DDIM steps (e.g., 50) give faster results with slight quality loss
3. **Quantization**: Lower QP values (5-20) give better quality but larger files
4. **Frame Size**: Ensure frame dimensions are multiples of 16 for optimal codec performance

## Troubleshooting

### Import Errors
If you get import errors for DCVC or MCVD modules, ensure:
1. Both repositories are cloned in accessible locations
2. Their paths are in your Python path or installed properly
3. All their dependencies are installed

### CUDA Out of Memory
- Reduce `--batch-size` to 1
- Reduce `--num-pred-frames`
- Use `--device cpu` (slower but uses less memory)

### File Not Found Errors
- Verify all model checkpoint paths are correct
- Ensure dataset paths point to the correct directories

## Original Notebooks

The repository includes the original Jupyter notebooks for reference:
- `KTH2_MCVD_DCVC1.ipynb`: KTH dataset experiments
- `BAIR_MCVD_DCVC1.ipynb`: BAIR dataset experiments
- `Graphs_Benchmarks_VC.ipynb`: Benchmarking and visualization

These notebooks contain the original working code that has been refactored into the organized structure.

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@inproceedings{dcvc_fm_2024,
  title={DCVC-FM: Deep Contextual Video Compression with Feature Modulation},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{mcvd_2022,
  title={MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

Please refer to the licenses of the respective components (DCVC, MCVD) for usage terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on GitHub.
