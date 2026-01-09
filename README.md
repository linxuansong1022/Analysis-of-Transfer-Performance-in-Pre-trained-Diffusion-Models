# Analysis of Transfer Performance in Pre-trained Diffusion Models

This repository explores the capabilities of **Pre-trained Diffusion Models (DDPM)** as frozen feature extractors for downstream tasks. It provides a comprehensive analysis of how features learned through generative denoising objectives can be transferred to discriminative tasks like **Image Classification** and **Signal Regression**.

## 🚀 Research Goals
- **Generative Pre-training**: Training 2D and 1D Diffusion models from scratch.
- **Feature Extraction**: Freezing the Diffusion UNet backbone to extract high-dimensional latent representations.
- **Transfer Learning**: Evaluating the performance of these features on downstream tasks (Classification/Regression) using lightweight heads (MLP, CNN, Attention).
- **Baselines**: Comparing results with Vision Transformer (ViT) architectures.

---

## 📁 Project Structure

The core implementation is located in the `Final_project/` directory, organized by task type and data dimensionality:

### 1. 2D Image Tasks (CIFAR-10)
*   **`Final_ddpmclassifier/`**: The main module for 2D diffusion.
    *   `Diffusion/`: Core DDPM logic and UNet model implementation.
    *   `Main.py`: Entry point for pre-training the generative diffusion model.
    *   `MainClassify.py`: Entry point for training the classifier on frozen features.
    *   `SampledImgs/`: Visualization of generated images during training.
    *   `DiffusionFreeGuidence/`: Implementation of Classifier-Free Guidance (CFG).

### 2. 1D Signal Tasks (Wave Echo Dataset)
*   **`Final_ddpmclassifier_1d_line/`**: 1D diffusion for signal classification.
    *   `Diffusion_1d/`: 1D UNet architecture and wave dataset loaders.
    *   `Main_1d.py`: Training the 1D generative model.
*   **`Final_reg` / `Final_regression`**: Modules focused on 1D signal regression.
    *   *Note: These folders contain parallel experiments for regression performance analysis.*

### 3. Baselines
*   **`FinalViT/`**: Contains the Vision Transformer implementation used as a performance baseline for the CIFAR-10 tasks.

---

## 🛠️ Key Components

### Diffusion Backbone
- **Model**: U-Net with Residual Blocks, Time Embeddings (Sinusoidal), and Self-Attention layers.
- **Process**: Implements standard Gaussian diffusion with a linear or cosine beta schedule.

### Downstream Heads
- **Classifier**: A simple MLP or Attention-based pooling layer added on top of the frozen UNet bottleneck/intermediate layers.
- **Regression**: Linear or CNN-based heads designed to map diffusion features to continuous values.

---

## 🚦 How to Use

### Step 1: Pre-training (Generative)
To train the diffusion model (e.g., for 2D images):
```bash
cd Final_project/Final_ddpmclassifier
python Main.py
```

### Step 2: Transfer Learning (Downstream)
Once the backbone is trained, freeze it and train the classifier:
```bash
python MainClassify.py
```

---

## 📊 Visualizations
Training progress and sampling results are automatically saved in the respective `SampledImgs/` folders within each sub-module. Check these for:
- Denoising quality at different timesteps.
- Final generated samples from the learned distribution.

---

## 📝 Notes on Current Status
- This project is under active organization.
- `Final_reg` and `Final_regression` are currently being consolidated into a unified regression suite.
