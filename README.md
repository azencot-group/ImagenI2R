# ImagenI2R : A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking

An implementation of time series to image generation using diffusion models, specifically designed for handling irregular time series.

## Overview

This project implements a novel approach for generating synthetic irregular time series data by:

1. **Time Series to Image Transformation**: Converting irregular time series to image representations using delay embedding
2. **Diffusion Model Training**: Training EDM (Elucidating the Design Space of Diffusion-Based Generative Models) on the image representations
3. **Time Series Transformer (TST)**: Using transformer-based encoder-decoder for time series reconstruction
4. **Synthetic Data Generation**: Generating new time series samples through the trained diffusion model

## Features

- **Irregular Time Series Support**: Handles missing data through NaN value propagation
- **Multiple Datasets**: Support for various time series datasets (electricity, energy, ETT, weather, stock, sine, mujoco)
- **Flexible Sequence Lengths**: Configurable sequence lengths (24, 96, 768)
- **Advanced Metrics**: Comprehensive evaluation using discriminative, predictive, FID, and correlation scores
- **EMA Support**: Exponential Moving Average for improved model stability
- **Neptune Integration**: Built-in experiment tracking with Neptune

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Data Setup

**Important**: You need to download all datasets from the provided Google Drive link and place them in the `data` folder.

1. Download all datasets from: [Google Drive Data Folder](https://drive.google.com/drive/folders/1fGGs6E5zlHxHiqqpGuQSESyEOt5GZpwg?usp=sharing)

2. Extract and place all files in the `data/` directory:
   ```
   data/
   ├── electricity.csv
   ├── energy.csv
   ├── ETTh1.csv
   ├── ETTh2.csv
   ├── ETTm1.csv
   ├── ETTm2.csv
   ├── stock.csv
   ├── weather.csv
   ├── mujoco_irregular.pt
   ├── mujoco_regular.pt
   └── mujoco.pt
   ```

## Usage

### Quick Start

Run with default configuration:

```bash
python run_irregular.py --config ./configs/seq_len_24/stock.yaml
```

### Configuration Options

The project supports three sequence lengths with pre-configured settings:

#### Sequence Length 24
```bash
python run_irregular.py --config ./configs/seq_len_24/[dataset_name].yaml
```

#### Sequence Length 96
```bash
python run_irregular.py --config ./configs/seq_len_96/[dataset_name].yaml
```

#### Sequence Length 768
```bash
python run_irregular.py --config ./configs/seq_len_768/[dataset_name].yaml
```

### Available Datasets

- `electricity` - Electricity consuming clients data
- `energy` - Energy consumption data
- `etth1`, `etth2` - Electricity Transformer Temperature (Hourly)
- `ettm1`, `ettm2` - Electricity Transformer Temperature (15-minute)
- `weather` - Weather measurements
- `stock` - Stock price data
- `sine` - Synthetic sine wave data
- `mujoco` - Physics simulation data

### Command Line Arguments

Key arguments you can override:

```bash
python run_irregular.py \
    --config ./configs/seq_len_96/electricity.yaml \
    --missing_rate 0.3 \
    --batch_size 32 \
    --epochs 1000 \
    --learning_rate 0.0001
```

#### Important Parameters

- `--missing_rate`: Proportion of missing data to simulate (default: 0.3)
- `--gaussian_noise_level`: Add noise to data (default: 0.0)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--neptune`: Enable Neptune logging (requires setup)

## Architecture

### Components

1. **DelayEmbedder** (`models/img_transformations.py`): Converts time series to images using delay embedding
2. **EDMPrecond** (`models/networks.py`): Diffusion model with EDM preconditioning
3. **TSTransformerEncoder** (`models/TST.py`): Transformer encoder for time series processing
4. **TST_Decoder** (`models/decoder.py`): GRU-based decoder for reconstruction

### Training Process

1. **Pre-training Phase**: Train TST encoder-decoder for time series reconstruction
2. **Diffusion Training**: Train diffusion model on image representations
3. **Joint Training**: Continue with both reconstruction and diffusion losses

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **Discriminative Score**: How well a classifier can distinguish real vs synthetic data
- **Predictive Score**: Prediction accuracy on downstream tasks
- **FID Score**: Fréchet Inception Distance adapted for time series
- **Correlation Score**: Pearson correlation between real and synthetic data

## Configuration Files

Configuration files are organized by sequence length:

```
configs/
├── seq_len_24/    # Short sequences
├── seq_len_96/    # Medium sequences  
└── seq_len_768/   # Long sequences
```

Each contains dataset-specific YAML files with optimized hyperparameters.

## Example Configuration

```yaml
# configs/seq_len_96/weather.yaml
epochs: 1000
batch_size: 32
learning_rate: 0.0001
dataset: weather
seq_len: 96
delay: 6
embedding: 16
img_resolution: 8
input_channels: 12
diffusion_steps: 18
```

## Results

Models are automatically saved when achieving best discriminative scores. Checkpoints include:

- Model weights (diffusion model, TST encoder/decoder)
- Optimizer states
- EMA weights (if enabled)
- Evaluation scores
- Training arguments

Saved models location: `./saved_models/seq_len_{length}/{dataset}/missing_rate_{rate}/`

## Neptune Integration

To use Neptune logging:

1. Set up Neptune account and project
2. Configure `utils/loggers/neptune/project.txt`
3. Run with `--neptune true`

## Dependencies

- `torch>=2.7.1` - Deep learning framework
- `numpy>=1.24.3` - Numerical computing
- `scipy>=1.11.2` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning utilities
- `omegaconf>=2.3.0` - Configuration management
- `Pillow>=10.0.0` - Image processing
- `tqdm>=4.66.1` - Progress bars

## Project Structure

```
├── configs/           # Configuration files by sequence length
├── data/             # Dataset files (download required)
├── metrics/          # Evaluation metrics implementation
├── models/           # Model architectures
├── utils/            # Utility functions and loggers
├── saved_models/     # Saved model checkpoints
├── run_irregular.py  # Main training script
└── requirements.txt  # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ts2img_irregular,
  title={Time Series to Image Generation for Irregular Data using Diffusion Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in config files
2. **Missing Data Files**: Ensure all datasets are downloaded and placed in `data/` folder
3. **Configuration Errors**: Check YAML syntax and parameter names
4. **Neptune Issues**: Verify Neptune setup and credentials

### Performance Tips

- Use GPU for training (automatically detected)
- Enable EMA for more stable results
- Adjust `logging_iter` for evaluation frequency
- Use appropriate `missing_rate` for your use case

## Support

For questions and issues, please:

1. Check the troubleshooting section
2. Review configuration files for examples
3. Open an issue with detailed error messages and configuration used
