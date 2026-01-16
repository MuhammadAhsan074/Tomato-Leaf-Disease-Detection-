# 🍅 Tomato Leaf Disease Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.25%25-success)
![Status](https://img.shields.io/badge/Status-Trained-green)

> **A Deep Learning solution utilizing Transfer Learning (MobileNetV2) to classify tomato leaf diseases with 90%+ accuracy.**

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Performance Metrics](#-performance-metrics)
3. [Dataset Details](#-dataset-details)
4. [Model Architecture](#-model-architecture)
5. [Installation & Usage](#-installation--usage)
6. [Future Improvements](#-future-improvements)
7. [License](#-license)

---

## 📖 Project Overview

This project implements a Convolutional Neural Network (CNN) to automatically diagnose tomato plant diseases from leaf images. By fine-tuning a pre-trained **MobileNetV2** model, the system effectively distinguishes between healthy leaves and various fungal diseases.

The model solves the problem of class imbalance using computed class weights and utilizes strong data augmentation to generalize well to unseen data.

---

## 📊 Performance Metrics

Based on the latest training session (40 Epochs), the model achieved the following performance on the **Validation Set**:

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **90.25%** |
| **Training Accuracy** | **91.12%** |
| **Macro F1-Score** | **0.89** |

### Classification Report (Per Class)

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Early Blight** | 0.96 | 0.72 | 0.82 | 113 |
| **Late Blight** | 0.84 | 0.94 | 0.89 | 166 |
| **Leaf Mold** | 0.90 | 0.90 | 0.90 | 146 |
| **Healthy** | 0.94 | 0.99 | 0.96 | 170 |

> *Observation: The model is exceptionally good at identifying Healthy leaves (99% Recall) and Leaf Mold.*

---

## 📂 Dataset Details

The dataset was processed from Google Drive with the following structure:

* **Total Images:** 2,979
* **Training Set:** 2,384 images (80%)
* **Validation Set:** 595 images (20%)
* **Classes (4):**
    1.  `Early_blight`
    2.  `Late_blight`
    3.  `Leaf_Mold`
    4.  `healthy`

---

## 🧠 Model Architecture

The solution uses **Transfer Learning** with the following pipeline:

1.  **Input Pipeline:** Resizing (224x224), Caching, Prefetching (`tf.data.AUTOTUNE`).
2.  **Augmentation:** RandomFlip, RandomRotation (0.25), RandomZoom (0.3), RandomContrast (0.3).
3.  **Base Model:** **MobileNetV2** (ImageNet weights).
    * *Fine-tuning:* Top 50 layers unfreezed for training.
4.  **Custom Head:**
    * GlobalAveragePooling2D
    * BatchNormalization
    * Dense (256 units, ReLU)
    * Dropout (0.5)
    * Output Dense (4 units, Softmax)

**Training Configuration:**
* **Optimizer:** Adam (Learning Rate: `1e-5`)
* **Loss Function:** Sparse Categorical Crossentropy
* **Callbacks:** EarlyStopping (Patience: 7), ReduceLROnPlateau.

---

## 🛠 Installation & Usage
 GuidePrerequisitesPython: 3.8, 3.9, or 3.10
 Hardware: NVIDIA GPU recommended for training (but runs on CPU).
 Memory: Minimum 8GB RAM


##  Future Improvements
Phase 1: Mobile Deployment (Q2 2026)
         Offline functionality for remote farms.
Action: Convert .keras model to TensorFlow Lite (.tflite).Platform: Develop an Android wrapper using Kotlin or Flutter.

Phase 2: Cloud API (Q3 2026)
        Objective: Scalable backend for multiple users.Action: Wrap the inference logic in a FastAPI container.Infrastructure: Deploy via Docker on AWS ECS.
Phase 3: Real-Time Drone Surveillance (2027)


### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/tomato-disease-classification.git](https://github.com/your-username/tomato-disease-classification.git)
cd tomato-disease-classification
