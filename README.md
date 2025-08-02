# Produce Type & Variation Classifier (Cauliflower, Plum, Corn & Ginger)

This project is a deep learning-based web application built using **Streamlit** that classifies the **type** and **variation** of agricultural produce, specifically **cauliflower**, **plum**, **corn** and **ginger**, using trained **ResNet50 models**.

---

## Project Structure

---

## Features

- Detect **type** of produce: Corn or Ginger
- Classify **variations** such as:
  - Cauliflower: Whole Head, Florets, Riced/Steaked
  - Plum: Whole, In a Bowl, Halved/Pitted
  - Corn: Husked, Unhusked, Kernels
  - Ginger: Whole, Sliced, In-Context
- Upload an image and get instant predictions
- Powered by fine-tuned ResNet50 models
- Simple browser-based UI with Streamlit

---

## Model Details

- **Architecture**: ResNet50 via Transfer Learning
- **Framework**: TensorFlow / Keras
- **Produce Type Model**: `produce_type_model.pkl`
- **Variation Model**: `produce_variation_model.pkl`
- **Input Size**: 150x150 RGB images
- **Trained on**: 12,000 images across all the four produces.

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
CV Engineers: Jacky He & Muhammad Waseem Thameem Ansari
Jacky He: Corn & Ginger Dataset, Model Development and Model Evaluation.
Muhammad Waseem Thameem Ansari: Cauliflower & Plum Dataset, Model Development and Model Evaluation.



