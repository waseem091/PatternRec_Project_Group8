# Produce Type & Variation Classifier (Corn + Ginger)

This project is a deep learning-based web application built using **Streamlit** that classifies the **type** and **variation** of agricultural produce, specifically **corn** and **ginger**, using trained **ResNet50 models**.

---

## Project Structure

---

## Features

- Detect **type** of produce: Corn or Ginger
- Classify **variations** such as:
  - Corn: Husked, Unhusked, Kernels
  - Ginger: Whole, Sliced, In-Context
- Upload an image and get instant predictions
- Powered by fine-tuned ResNet50 models
- Simple browser-based UI with Streamlit

---

## Model Details

- **Architecture**: ResNet50 via Transfer Learning
- **Framework**: TensorFlow / Keras
- **Produce Type Model**: `produce_type_resnet50.h5`
- **Variation Model**: `produce_variation_resnet.h5`
- **Input Size**: 224x224 RGB images
- **Trained on**: 6000+ images across both crop categories

---

## How to Run Locally

> Requires Python 3.8 or newer.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/PatternRec_Project_Group8.git
cd PatternRec_Project_Group8/streamlit_app
pip install streamlit tensorflow numpy pillow matplotlib
streamlit run app.py
```

## Contributors
CV Engineer: Jacky He & Muhammad Waseem 



