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




📊 Performance Results
Overall Metrics
Metric	Value
Accuracy	92%
Precision	93%
Recall	91%
F1-Score	92%
Class-wise Performance
Class	Precision	Recall	F1-Score
Early Blight	94%	80%	86%
Late Blight	86%	93%	89%
Leaf Mold	94%	93%	94%
Healthy	97%	99%	98%

Observation:
Validation accuracy closely follows training accuracy, indicating no overfitting.

🚀 Installation & Usage
Prerequisites

Python 3.8+

TensorFlow 2.10+

8GB+ RAM (GPU recommended)

Quick Start
git clone https://github.com/yourusername/tomato-leaf-disease-classification.git
cd tomato-leaf-disease-classification
pip install -r requirements.txt
python train_model.py

Prediction Example
python predict.py --image test_leaf.jpg

📱 Deployment

The trained model can be exported in multiple formats:

model.save("tomato_leaf_disease_model_v2.keras")
pickle.dump(model, open("tomato_leaf_disease_model.pkl", "wb"))


Designed for:

TensorFlow Lite conversion

Mobile and edge deployment

Offline inference

🔮 Future Work

Dataset expansion under varied field conditions

EfficientNet and Vision Transformer experiments

TensorFlow Lite mobile app

Multi-crop disease classification

IoT sensor and weather data integration

📁 Project Structure
tomato-leaf-disease-classification/
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── Early_blight/
│   ├── Late_blight/
│   ├── Leaf_Mold/
│   └── healthy/
├── models/
│   ├── tomato_model_v2.keras
│   └── tomato_model.pkl
└── plots/
    ├── training_history.png
    └── confusion_matrix.png

📄 License

This project is licensed under the MIT License.

⭐ If you find this project useful, please consider giving it a star. ⭐
