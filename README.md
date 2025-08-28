# S3F-Net
S³F-Net: A Multi-Modal Approach to Medical Image Classification via Spatial-Spectral Summarizer Fusion Network

**Official TensorFlow/Keras implementation for the paper: "S³F-Net: A Multi-Modal Approach to Medical Image Classification via Spatial-Spectral Summarizer Fusion Network"**

> **Abstract:** Standard Convolutional Neural Networks (CNNs) have become a cornerstone of medical image analysis due to their proficiency in learning hierarchical spatial features. However, this focus on a single domain is inefficient at capturing global, holistic patterns and fails to explicitly model an image’s frequency- domain characteristics. To address these challenges, we propose the Spatial-Spectral Summarizer Fusion Network (S³F-Net), a dual-branch framework that learns from both spatial and spectral representations simultaneously. The S³F-Net integrates a deep spatial CNN for local morphology with our proposed SpectraNet branch. A key innovation within SpectraNet is the SpectralFilter Layer, which leverages the Convolution Theorem by applying a bank of learnable filters directly to an image’s Fourier spectrum via a parameter-efficient element-wise multiplication. This allows the SpectraNet to attain a global receptive field from its first layer, with its output being distilled by a lightweight ”funnel” head. We conduct a comprehensive investigation across four diverse medical imaging datasets to demonstrate S³F-Net’s efficacy and show that the optimal fusion strategy is task-dependent. With a powerful Bilinear Fusion, S³F-Net achieves a state-of-the- art competitive accuracy of 98.76% on the BRISC2025 brain tumor dataset. In contrast, a simpler Concatenation Fusion proves superior on the texture-dominant Chest X-Ray Pneumonia dataset, achieving 93.11% accuracy, also a result surpassing many top-performing, much deeper models. In all evaluated cases, the S³F-Net framework significantly outperforms its strong spatial- only baseline, demonstrating that our dual-domain approach is a powerful and generalizable paradigm for medical image analysis.

---

## Architecture Overview

The S³F-Net is an asymmetric two-tower architecture. It consists of:
1.  A deep, conventional **Spatial Branch** (CNN) that acts as a powerful expert on local, morphological features.
2.  Our novel, shallow, and parameter-efficient **SpectraNet Branch** that acts as an expert on global, textural, and frequency-based features.

The final, refined feature vectors from these two independent experts are then combined using a task-dependent fusion strategy to produce the final classification.

![S³F-Net Architecture Diagram](figures/s3fnet_architecture.png)

---

## Key Results

A central finding of this research is that the optimal fusion strategy is **task-dependent**. Our S³F-Net framework consistently and significantly outperforms its strong spatial-only baseline across all four datasets.

| Dataset | Baseline (Spatial-Only) | S³F-Net (Best Fusion) | Performance Gain | Best Fusion Method |
| :--- | :--- | :--- | :--- | :--- |
| **BRISC2025 (MRI)** | 98.27% Acc. | **98.76% Acc.** | +0.49% | **Bilinear Fusion** |
| **Chest X-Ray** | 87.98% Acc. | **93.11% Acc.** | **+5.13%** | **Concatenation** |
| **BUSI (Ultrasound)** | 84.30% Acc. | **87.82% Acc.** | +3.52% | **Concatenation** |
| **HAM10000** | 0.689 W. F1 | **0.704 W. F1** | +1.5% | **Bilinear Fusion** |

The Bilinear Fusion excels on feature-complex tasks, while the simpler Concatenation Fusion is superior for texture-dominant modalities.

---

## Setup and Installation

This project is built using TensorFlow 2.18.

**1. Clone the repository:**
```bash
git clone https://github.com/Saiful185/S3F-Net.git
cd S3F-Net
```

**2. Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
Key dependencies include: `tensorflow`, `pandas`, `opencv-python`, `scikit-learn`, `seaborn`.

---

## Dataset Preparation

The experiments are run on four publicly available datasets. For fast I/O, it is highly recommended to download the datasets, zip them, upload the zip file to your Google Drive, and then use the unzipping cells in the Colab notebooks.

#### 1. HAM10000
- **Download:** From the [Kaggle dataset page](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification) or the ISIC archive.
  
#### 2. BRISC2025
- **Download:** From the [Kaggle dataset page](https://www.kaggle.com/datasets/briscdataset/brisc2025).

#### 3. Chest X-Ray (Pneumonia)
- **Download:** From the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

#### 4. BUSI
- **Download:** From the [Kaggle dataset page](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

## Usage: Running the Experiments

The code is organized into Jupyter/Colab notebooks (`.ipynb`) for each key experiment.

1.  **Open a notebook** (e.g., `S3F-Net_Bilinear_SN1_BRISC2025.ipynb`).
2.  **Update the paths** in the first few cells to point to your dataset's location (either on Google Drive for unzipping or a local path).
3.  **Run the cells sequentially** to perform data setup, model training, and final evaluation.

---

## Citation

If you find this work useful in your research, please consider citing our paper:

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
