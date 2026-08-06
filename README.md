# Potato Leaf Disease Detection with Explainable AI (XAI)

## Overview

This project uses Deep Learning and Explainable AI (XAI) techniques to detect potato leaf diseases from images. The system classifies potato leaves into Healthy, Early Blight, and Late Blight categories and provides visual explanations for model predictions using Grad-CAM, LIME, and SHAP.

## Features

* Potato leaf disease classification using CNN
* Detection of:

  * Healthy
  * Early Blight
  * Late Blight
* Explainable AI (XAI) integration
* Grad-CAM heatmap visualization
* LIME local explanations
* SHAP feature importance analysis
* Confidence score prediction

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* LIME
* SHAP
* Scikit-learn

## Project Structure

Potato_Disease_XAI/
│
├── dataset/
├── models/
│   └── potato_disease_model.h5
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── lime_explain.py
│   └── shap_explain.py
│
├── outputs/
│   ├── gradcam/
│   ├── lime/
│   └── shap/
│
├── requirements.txt
├── README.md
└── .gitignore

## Explainable AI Techniques

### Grad-CAM

Generates heatmaps highlighting image regions that most influence the CNN prediction.

### LIME (Local Interpretable Model-Agnostic Explanations)

Provides local explanations by identifying image segments that contribute positively or negatively to a prediction.

### SHAP (SHapley Additive exPlanations)

Computes feature importance values based on cooperative game theory, showing how different image regions contribute to disease classification.

## Dataset

PlantVillage Potato Leaf Disease Dataset

Classes:

* Healthy
* Early Blight
* Late Blight

## Results

* Accurate disease classification
* Visual explanation using Grad-CAM
* Local interpretation using LIME
* Feature attribution using SHAP

## Future Scope

* Real-time disease detection
* Mobile application deployment
* Support for additional crop diseases
* Cloud-based monitoring system

## Author

Tanishka Pangavhane |
CSE
