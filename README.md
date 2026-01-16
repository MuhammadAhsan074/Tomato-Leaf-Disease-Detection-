# 🍅 Tomato Leaf Disease Classification (90% Accuracy)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A Deep Learning solution for automated diagnosis of tomato plant diseases using MobileNetV2 and Transfer Learning.**

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Future Scope](#-future-scope)

---

## 🔍 Project Overview

Agriculture faces significant losses due to plant diseases that are often detected too late. This project provides an automated, AI-driven solution to classify **10 different classes** of tomato leaf diseases (including healthy leaves) with high precision.

By leveraging **Transfer Learning** with the **MobileNetV2** architecture, the model achieves robust performance even with limited computational resources, making it suitable for deployment in edge devices or mobile applications.

### 🎯 Objectives
* Build a robust image classifier using **TensorFlow/Keras**.
* Achieve a validation accuracy target of **95%**.
* Develop a user-friendly web interface using **Streamlit**.

---

## 🌟 Key Features

* **Advanced Pre-processing:** Implements data augmentation (rotation, zoom, contrast) to handle real-world image variability.
* **Transfer Learning:** Uses a pre-trained MobileNetV2 model (fine-tuned) for faster convergence and higher accuracy.
* **Real-time Prediction:** Instant diagnosis via the web interface.
* **Confidence Scores:** Displays the probability percentage of the diagnosis to aid decision-making.

---

## 📂 Dataset

The model is trained on a labeled dataset containing images of tomato leaves categorized into the following 10 classes:

1.  Tomato___Bacterial_spot
2.  Tomato___Early_blight
3.  Tomato___Late_blight
4.  Tomato___Leaf_Mold
5.  Tomato___Septoria_leaf_spot
6.  Tomato___Spider_mites Two-spotted
