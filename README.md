🍅 Professional Tomato Leaf Disease Classification (v2.0)










📌 Project Overview

This repository contains a professional deep learning pipeline for automated tomato leaf disease classification using transfer learning with MobileNetV2.
The system is designed for precision agriculture, enabling early disease detection to reduce crop loss and improve treatment efficiency.

Version 2.0 introduces a production-grade architecture with improved robustness, interpretability, and deployment readiness.

Key Highlights

Architecture: MobileNetV2 pretrained on ImageNet with fine-tuning

Accuracy: 92% validation accuracy

Robustness: Strong data augmentation for real-field conditions

Imbalance Handling: Automated class weighting

Explainability: Confusion Matrix, Learning Curves, F1-Score analysis

Deployment: Mobile-ready and exportable (.keras, .pkl)

📂 Dataset

The model is trained on a curated Tomato Leaf Disease Dataset consisting of four classes:

Healthy

Early Blight (fungal infection)

Late Blight (water mold infection)

Leaf Mold (fungal pathogen)

Dataset Configuration

Total Images: 2,979

Input Resolution: 224 × 224

Batch Size: 32

Train / Validation Split: 80% / 20%

Class Distribution (Approx.)
Class	Images
Early Blight	~750
Late Blight	~1100
Leaf Mold	~850
Healthy	~280
🏗️ Model Architecture

The pipeline leverages MobileNetV2 as an efficient feature extractor optimized for edge and mobile deployment.

Architecture Flow

Input Layer: (224, 224, 3)

Data Augmentation:

Random Flip

Random Rotation (0.25)

Random Zoom (0.3)

Random Contrast (0.3)

Random Brightness (0.2)

Base Model: MobileNetV2 (ImageNet pretrained)

Bottom layers frozen

Top layers fine-tuned

Global Average Pooling

Classifier Head:

Batch Normalization

Dense (ReLU)

Dropout (0.5)

Dense (Softmax, 4 classes)

Total Parameters: ~2.26M
Trainable Parameters: ~1.85M

⚙️ Training Configuration

Optimizer: Adam

Learning Rate: 1e-5

Loss Function: Sparse Categorical Crossentropy

Epochs: 40

Callbacks:

Early Stopping (patience = 8)

ReduceLROnPlateau (factor = 0.3)

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
