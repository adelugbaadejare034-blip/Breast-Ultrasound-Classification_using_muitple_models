# Breast Ultrasound Image Classification using Deep Learning & ML 🎗️

This repository hosts a comparative study between Deep Learning and Classical Machine Learning approaches for the automated classification of breast ultrasound images. The project utilizes the **Breast Ultrasound Images Dataset (BUSI)** to classify scans into three categories: **Benign, Malignant, and Normal**.

## 📊 Project Overview

Accurate diagnosis of breast cancer from ultrasound imagery is a critical challenge in medical imaging. This project implements and evaluates two distinct modeling approaches to address this:

1.  **Deep Learning (DL):** A Convolutional Neural Network (CNN) based on **EfficientNet-B0** using Transfer Learning.
2.  **Machine Learning (ML):** A traditional **Random Forest Classifier** combined with Principal Component Analysis (PCA) for dimensionality reduction.

## 📂 Dataset
* **Source:** [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)
* **Classes:** Benign, Malignant, Normal.
* **Challenges:** The original dataset is small and imbalanced, requiring significant preprocessing and augmentation.

## 🧠 Methodology

### 1. Data Preprocessing & Augmentation
Medical datasets often suffer from class imbalance. To ensure robust training, we implemented a custom augmentation pipeline using `TensorFlow` to balance the classes.
* **Target:** Generated 4,000 augmented images per class (12,000 total).
* **Techniques Applied:**
    * Random Horizontal/Vertical Flips
    * Random Rotation (90°, 180°, 270°)
    * Random Cropping
* **Resizing:**
    * **DL Model:** Resized to `224x224` pixels (standard input for EfficientNet).
    * **ML Model:** Resized to `64x64` pixels (to reduce dimensionality for PCA).

### 2. Deep Learning Model: EfficientNet-B0
We utilized Transfer Learning with **EfficientNet-B0**, a state-of-the-art CNN architecture known for balancing accuracy and efficiency.
* **Architecture:** EfficientNet-B0 pre-trained on ImageNet.
* **Modifications:** The final classification head was replaced to output 3 classes (Benign, Malignant, Normal).
* **Loss Function:** CrossEntropyLoss.
* **Optimizer:** Adam (Learning Rate = 1e-4).
* **Training:** 25 Epochs.

### 3. Machine Learning Model: Random Forest
As a baseline, we implemented a classical ML pipeline using `scikit-learn`.
* **Feature Extraction:** Images were flattened into 1D vectors.
* **Dimensionality Reduction:** **Principal Component Analysis (PCA)** was applied to reduce the feature space to **100 components**, preserving the most significant variance while reducing noise.
* **Classifier:** **Random Forest** with the following hyperparameters:
    * `n_estimators`: 100 (Number of trees)
    * `class_weight`: 'balanced' (To handle any remaining class imbalance)
    * `random_state`: 42 (For reproducibility)

---

## 📈 Results & Performance

The Deep Learning approach significantly outperformed the Classical ML model, demonstrating the superior ability of CNNs to capture spatial hierarchies in medical images.

### 🏆 Model Comparison Summary

| Metric | EfficientNet-B0 (DL) | Random Forest (ML) |
| :--- | :--- | :--- |
| **Accuracy** | **95.75%** | 58.20% |
| **F1-Score (Weighted)** | **0.96** | 0.58 |
| **AUC-ROC** | **0.99** | 0.75 |

### 📝 Detailed Classification Reports

#### 1. EfficientNet-B0 (Deep Learning)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.98 | 0.90 | 0.94 | 807 |
| **Malignant** | 0.94 | 0.98 | 0.96 | 793 |
| **Normal** | 0.96 | 0.99 | 0.97 | 800 |

*Analysis: The DL model achieved exceptional sensitivity (Recall) for Malignant cases (98%), which is crucial for medical diagnosis to avoid false negatives.*

#### 2. Random Forest (Machine Learning)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Benign** | 0.55 | 0.58 | 0.57 | 800 |
| **Malignant** | 0.62 | 0.55 | 0.58 | 800 |
| **Normal** | 0.58 | 0.61 | 0.60 | 800 |

*Analysis: The Random Forest model struggled to differentiate between classes, achieving only 55% recall for Malignant cases, highlighting the limitations of pixel-based features for complex ultrasound analysis.*

---

## 🚀 Comparison Plots
*(Place your generated plots here)*

* **Confusion Matrices:** Visualizes the true vs. predicted labels for both models.
* **Loss Curves:** Shows the training and validation loss convergence for the DL model.
* **Metric Comparison Bar Chart:** A side-by-side comparison of Accuracy, Precision, and Recall.

---

## 🛠️ Requirements
To run the code, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision tensorflow scikit-image tqdm
