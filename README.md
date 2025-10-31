# Analysis of Transfer Performance in Pre-trained Diffusion Models

This repository contains the code and findings for the undergraduate thesis: **"Analysis of Transfer Performance in Pre-trained Diffusion Models"**.

## 🚀 Project Overview

This research explores the capability of pre-trained diffusion models (like DDPM) to act as powerful general-purpose feature extractors for downstream discriminative tasks.

Instead of training complex models from scratch, this project utilizes a **frozen backbone** strategy:
1.  A diffusion model is pre-trained on a generative task (denoising).
2.  The entire pre-trained network is then **frozen**.
3.  Lightweight classification or regression heads are attached to the model's intermediate layers to extract features.
4.  Only these small heads are trained on the downstream task, making the process highly efficient.

---

## 📊 Key Results

This method was validated on two distinct tasks: 2D image classification and 1D signal regression.

### 1. Image Classification (on CIFAR-10)

* **Result:** Our method (DDPM + Multi-Head Attention Head) achieved **89.71%** accuracy.
* **Comparison:** This significantly outperformed the self-supervised Vision Transformer (ViT) baseline, which scored 85.65%.
* **Optimal Parameters:** The best-performing features were extracted at a moderate noise level (**t=30**) from a late network layer (**block 24**).

### 2. 1D Signal Regression (Wave Height Prediction)

* **Result:** Our method (1D-DDPM + Single Linear Layer) achieved a Mean Squared Error (MSE) of **0.0041**.
* **Comparison:** This was more than twice as accurate as the baseline (MSE 0.0090) and also outperformed a 3-layer MLP (MSE 0.0052).
* **Optimal Parameters:** The best-performing features were extracted at a low noise level (**t=10**) from a middle network layer (**block 16**).

## 💡 Conclusion

The features learned by diffusion models during their generative training are rich in discriminative information. This study demonstrates that these models can be successfully and efficiently transferred to downstream tasks, acting as powerful, pre-trained feature extractors.

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@thesis{song2025analysis,
  title  = {Analysis of Transfer Performance in Pre-trained Diffusion Models (预训练扩散模型的迁移性能分析)},
  author = {Song, Linxuan (宋林璇)},
  year   = {2025}
}