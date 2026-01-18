# 🍅 Professional Tomato Leaf Disease Classification (v2.0)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Success-green)
![Accuracy](https://img.shields.io/badge/Accuracy-92%25-brightgreen)

## 📌 Project Overview
This project implements a robust deep learning pipeline for detecting diseases in tomato leaves using **Transfer Learning with MobileNetV2**. It is designed to assist in precision agriculture by automating the identification of plant stress.

**Version 2.0** introduces significant improvements over standard CNN approaches:
* **Architecture:** MobileNetV2 (Pre-trained on ImageNet) with fine-tuning.
* **Robustness:** Strong data augmentation (Zoom, Rotation, Contrast, Brightness) to handle field conditions.
* **Balance:** Automated Class Weight computation to handle dataset imbalance.
* **Visualization:** Comprehensive reporting including Confusion Matrices, Learning Curves, and F1-Score distributions.

## 📂 Dataset
The model is trained on the **Tomato Leaf Disease Dataset** containing 4 distinct classes:

1.  **Healthy**
2.  **Early Blight** (Fungal infection)
3.  **Late Blight** (Water mold infection)
4.  **Leaf Mold** (Fungal pathogen)

* **Input Size:** 224x224 pixels
* **Batch Size:** 32
* **Train/Val Split:** 80% / 20%

## 🏗️ Model Architecture
The pipeline utilizes **MobileNetV2** as the feature extractor, which is highly efficient for mobile and edge deployment.

1.  **Input Layer:** `(224, 224, 3)`
2.  **Data Augmentation:** RandomFlip, RandomRotation (0.25), RandomZoom (0.3), RandomContrast (0.3), RandomBrightness (0.2).
3.  **Base Model:** MobileNetV2 (Frozen bottom layers, Fine-tuned top 50 layers).
4.  **Global Average Pooling:** Reduces spatial dimensions.
5.  **Classifier Head:**
    * Batch Normalization
    * Dense (256, ReLU)
    * Dropout (0.5) to prevent overfitting
    * Output Dense (Softmax, 4 classes)

## 📊 Performance Results
