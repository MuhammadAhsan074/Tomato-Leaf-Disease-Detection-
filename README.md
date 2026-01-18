
🍅 Professional Tomato Leaf Disease Classification (v2.0)


📖 Introduction
This project implements a robust deep learning pipeline for the automated classification of tomato leaf diseases. Utilizing Transfer Learning with MobileNetV2, the system is designed to identify four distinct classes of plant health conditions. Version 2.0 introduces advanced data augmentation, class balancing, and professional-grade visualization to ensure high reliability and interpretability.

The model achieves a 92% weighted F1-score, making it suitable for agricultural deployment and research analysis.

📑 Table of Contents
Project Features

Dataset Details

Methodology & Architecture

Installation & Requirements

Usage

Visualizations

Results & Evaluation

Conclusion

Project Data & Metrics (Tables)

🚀 Project Features
Architecture: Fine-tuned MobileNetV2 (Pre-trained on ImageNet).

Robustness: Implements strong data augmentation (Flip, Rotation, Zoom, Contrast, Brightness).

Optimization: Uses class_weight="balanced" to handle dataset imbalance.

Performance: Implements Learning Rate Decay and Early Stopping.

Serialization: Saves models in both .keras (Professional) and .pkl (Legacy) formats.

Analytics: Generates 6 different visualization graphs including Confusion Matrix and F1-Score distributions.

📂 Dataset Details
The dataset consists of tomato leaf images divided into four classes. The pipeline automatically handles corrupt images and splits the data into training and validation sets.

Source Path: /content/drive/MyDrive/Tomato datasets

Total Images: 2,979

Image Dimensions: 224×224 pixels

Classes: Early_blight, Late_blight, Leaf_Mold, healthy

🧠 Methodology & Architecture
1. Data Pipeline
Preprocessing: Rescaling (1./255).

Augmentation: Random transformations applied dynamically during training to prevent overfitting.

Performance: Uses tf.data.AUTOTUNE for caching and prefetching.

2. Model Structure
Base: MobileNetV2 (Top layers removed, weights frozen initially).

Fine-Tuning: The top 50 layers of the base model were un-frozen to adapt to specific leaf features.

Head: Global Average Pooling → Batch Normalization → Dense (256, ReLU) → Dropout (0.5) → Output (Softmax).

💻 Installation & Requirements
To run this project, ensure you have the following libraries installed:

Bash

pip install tensorflow matplotlib seaborn pandas scikit-learn
Environment:

Designed for Google Colab (requires Google Drive mounting).

Hardware Acceleration: GPU (Recommended).

🛠 Usage
Mount Drive: Ensure your dataset is uploaded to Google Drive.

Configure Paths: Update DATASET_PATH in the script if your folder structure differs.

Run Training: Execute the script. The pipeline will:

Load and split data.

Visualize class distribution.

Train the model for up to 40 epochs.

Generate evaluation metrics and save the model.

📊 Visualizations
The system generates the following insights:

Class Distribution: Bar chart showing data balance.

Accuracy Curve: Training vs. Validation accuracy over epochs.

Loss Curve: Cross-entropy loss reduction over time.

Learning Rate Schedule: Visualization of LR decay triggers.

Confusion Matrix: Heatmap of True vs. Predicted labels.

Per-Class F1 Score: Bar chart showing performance per disease type.

🏆 Results & Evaluation
The model converged successfully after 40 epochs with highly stable metrics.

Final Validation Accuracy: 91.93%

Final Validation Loss: 0.2582

Best Performing Class: healthy (98% F1-Score)

Classification Insights
The model distinguishes healthy and Leaf_Mold with near-perfect precision. Early_blight presents the highest difficulty but still achieves a respectable 86% F1-score.

📝 Conclusion
This project demonstrates a highly effective application of transfer learning in agriculture. By leveraging MobileNetV2 and balancing class weights, the model overcomes common challenges like data imbalance and overfitting. The resulting system is lightweight, accurate, and ready for deployment in mobile or edge devices for real-time plant disease detection.

📈 Project Data & Metrics (Appendices)
Below is a detailed breakdown of the project specifications and results.

Table 1: Project Overview
Attribute	Details
Project Name	Tomato Leaf Disease Classification v2.0
Model Architecture	MobileNetV2 (Transfer Learning)
Input Shape	(224,224,3)
Framework	TensorFlow / Keras
Execution Environment	Google Colab

Export to Sheets

Table 2: Dataset Statistics
Dataset Partition	Number of Images	Percentage
Training Set	2,384	80%
Validation Set	595	20%
Total Images	2,979	100%
Number of Classes	4	-

Export to Sheets

Table 3: Training Configuration
Parameter	Value
Epochs	40
Batch Size	32
Initial Learning Rate	1e 
−5
 
Optimizer	Adam
Loss Function	Sparse Categorical Crossentropy

Export to Sheets

Table 4: Final Training Metrics (Epoch 40)
Metric	Result
Training Accuracy	90.79%
Training Loss	0.2344
Validation Accuracy	91.93%
Validation Loss	0.2582

Export to Sheets

Table 5: Detailed Classification Report
Class Name	Precision	Recall	F1-Score	Support
Early_blight	0.94	0.80	0.86	113
Late_blight	0.86	0.93	0.89	166
Leaf_Mold	0.94	0.93	0.94	146
healthy	0.97	0.99	0.98	170
Weighted Avg	0.92	0.92	0.92	595

Export to Sheets

Table 6: Optimization Techniques Used
Technique	Purpose
Class Weights	Balanced training for under-represented classes
Data Augmentation	RandomFlip, Rotation, Zoom, Contrast, Brightness
Early Stopping	Prevent overfitting (Patience = 8)
ReduceLROnPlateau	Fine-tune learning rate when loss stagnates
GlobalAvgPooling	Reduce parameters and prevent overfitting

Export to Sheets

Table 7: Generated Output Files
File Name	Format	Description
tomato_leaf_disease_model_professional_v2.keras	Keras	Full model (Architecture + Weights)
tomato_leaf_disease_model.pkl	Pickle	Serialized model object
history object	Python Dict	Contains loss/accuracy logs for plotting
