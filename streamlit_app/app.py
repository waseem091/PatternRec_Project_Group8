import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Utility Functions
# -------------------------

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, image_tensor, class_names):
    preds = model.predict(image_tensor)[0]
    top_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return top_class, confidence

# -------------------------
# UI & Layout
# -------------------------

st.set_page_config(page_title="Produce Classifier", layout="centered")

st.title("🌾 Produce Classifier App")
st.caption("Detecting the type and variation of produce (Corn 🌽 & Ginger 🌱) using CNNs")

# Load models
produce_model = load_model("produce_type_resnet50.h5")
variation_model = load_model("produce_variation_resnet50.h5")

# Class labels
produce_labels = ["Corn", "Ginger"]
variation_labels = [
    "Corn - Whole", "Corn - Sliced", "Corn - In-Context",
    "Ginger - Whole", "Ginger - Sliced", "Ginger - In-Context"
]

# Tabs for classifier switching
tab1, tab2 = st.tabs(["🧪 Produce Type Classifier", "🔍 Variation Classifier"])

# -------------------------
# Tab 1: Produce Type Classifier
# -------------------------
with tab1:
    st.subheader("🧪 What kind of produce is this?")
    uploaded_file = st.file_uploader("Upload an image of Corn or Ginger", type=["jpg", "png", "jpeg"], key="produce_upload")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Your Image", use_column_width=True)

        tensor = preprocess_image(img)
        pred_class, confidence = predict(produce_model, tensor, produce_labels)

        st.success(f"🔎 This looks like **{pred_class}** with **{confidence:.2f}%** confidence.")

# -------------------------
# Tab 2: Variation Classifier
# -------------------------
with tab2:
    st.subheader("🔍 What variation of produce is this?")
    uploaded_file2 = st.file_uploader("Upload a variation image (Whole / Sliced / In-Context)", type=["jpg", "png", "jpeg"], key="variation_upload")

    if uploaded_file2:
        img2 = Image.open(uploaded_file2).convert("RGB")
        st.image(img2, caption="Your Image", use_column_width=True)

        tensor2 = preprocess_image(img2)
        pred_class2, confidence2 = predict(variation_model, tensor2, variation_labels)

        st.success(f"🧠 Predicted variation: **{pred_class2}** with **{confidence2:.2f}%** confidence.")
