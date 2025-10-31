# DDPM Classifier

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation and classification tasks. The implementation includes both the core diffusion model and a classification head that can be trained on top of the pre-trained diffusion model.

## Features

- **Diffusion Model**: Implements a U-Net based architecture for image generation using the DDPM framework
- **Classification Capability**: Can be used for image classification by adding different types of classification heads:
  - Fully Connected (FC) head
  - Multi-Layer Perceptron (MLP) head
  - Attention-based head
  - Convolutional Neural Network (CNN) head
- **Training Pipeline**: Includes both training the diffusion model and fine-tuning the classification head
- **CIFAR-10 Support**: Currently configured to work with the CIFAR-10 dataset

## Architecture

The model consists of several key components:

1. **U-Net Backbone**: The main architecture for the diffusion model, featuring:
   - Time embedding layers
   - Downsampling and upsampling blocks
   - Residual blocks with attention mechanisms
   - Skip connections

2. **Diffusion Process**: Implements the forward and reverse diffusion processes with:
   - Customizable number of timesteps (T)
   - Configurable beta schedule
   - Noise prediction and denoising

3. **Classification Heads**: Multiple options for the classification layer:
   - FC: Simple fully connected layer
   - MLP: Multi-layer perceptron with batch normalization
   - Attention: Self-attention based classification
   - CNN: Convolutional neural network based classification

## Requirements

- PyTorch
- CUDA-capable GPU (required for training)
- Python 3.x
- torchvision
- tqdm

## Usage

### Training the Diffusion Model

```python
python Main.py
```

### Training the Classifier

```python
python MainClassify.py
```

## Configuration

The model can be configured through the `modelConfig` dictionary in `Main.py` or `MainClassify.py`. Key parameters include:

- `state`: "train" or "eval"
- `epoch`: Number of training epochs
- `batch_size`: Training batch size
- `T`: Number of diffusion timesteps
- `channel`: Base number of channels
- `channel_mult`: Channel multiplier for different scales
- `attn`: Attention layer positions
- `num_res_blocks`: Number of residual blocks
- `dropout`: Dropout rate
- `lr`: Learning rate
- `device`: Training device (e.g., "cuda:0")

## Project Structure

```
.
├── Diffusion/
│   ├── __init__.py
│   ├── Diffusion.py      # Diffusion model implementation
│   ├── Model.py          # U-Net architecture
│   └── Train.py          # Training and evaluation code
├── Main.py               # Main script for diffusion model training
├── MainClassify.py       # Main script for classifier training
└── README.md
```

## Notes

- The model requires a CUDA-capable GPU for training
- Pre-trained weights can be loaded for fine-tuning or evaluation
- The default configuration is optimized for CIFAR-10 images (32x32) 



