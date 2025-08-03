import streamlit as st
import tensorflow as tf
import numpy as np
import os
import base64
from PIL import Image

# -------------------------
# Utility Functions
# -------------------------

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(img, target_size):
    """Resize image to target size and normalize to [0,1]."""
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
st.caption("Detecting the type and variation of produce (Cauliflower 🥦, Plum 🟣, Corn 🌽 & Ginger 🫚) using CNNs")

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "h5-files")

# Produce type model
produce_model = load_model(os.path.join(MODEL_DIR, "produce_type_model.h5"))

# Load separate variation models for each produce type
variation_models = {
    "Cauliflower": load_model(os.path.join(MODEL_DIR, "Cauliflower_variation_model.h5")),
    "Plum": load_model(os.path.join(MODEL_DIR, "Plum_variation_model.h5")),
    "Corn": load_model(os.path.join(MODEL_DIR, "Corn_variation_model.h5")),
    "Ginger": load_model(os.path.join(MODEL_DIR, "Ginger_variation_model.h5")),
}

# Global class labels
produce_labels = ["Cauliflower", "Plum", "Corn", "Ginger"]

variation_labels_map = {
    "Cauliflower": ["Cauliflower - Whole Head", "Cauliflower - Florets", "Cauliflower - Riced (or) Steaked"],
    "Plum": ["Plum - Whole", "Plum - In a Bowl", "Plum - Halved (or) Pitted"],
    "Corn": ["Corn - Husked", "Corn - Kernels", "Corn - Un-husked"],
    "Ginger": ["Ginger - Broken (or) Peeled", "Ginger - Minced (or) Sliced", "Whole Hand"],
}

# Sidebar navigation
st.sidebar.title("Produce Classifier")
selection = st.sidebar.radio(
    "Go to",
    [
        "Produce Type Classifier",
        "Produce Variation Classifier",
        "Scientific Report",
    ],
)

# Hide radio button circles in sidebar
hide_radio_style = """
    <style>
    div[data-testid='stSidebar'] div[role='radiogroup'] label div:first-child {
        display: none;
    }
    div[data-testid='stSidebar'] div[role='radiogroup'] label{
        padding-left: 0;
        cursor: pointer;
    }
    </style>
"""
st.markdown(hide_radio_style, unsafe_allow_html=True)

# -------------------------
# Page 1: Produce Type Classifier
# -------------------------
if selection == "Produce Type Classifier":
    st.subheader("🧪 What kind of produce is this?")
    uploaded_file = st.file_uploader(
        "Upload an image of Cauliflower, Plum, Corn or Ginger", type=["jpg", "png", "jpeg"], key="produce_upload"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Your Image", use_container_width=True)

        # Preprocess specifically for the produce type model
        tensor = preprocess_image(img, target_size=produce_model.input_shape[1:3])
        pred_class, confidence = predict(produce_model, tensor, produce_labels)

        st.success(
            f"🔎 This looks like **{pred_class}** with **{confidence:.2f}%** confidence."
        )

# -------------------------
# Page 2: Produce Variation Classifier
# -------------------------
elif selection == "Produce Variation Classifier":
    st.subheader("🔍 What variation of produce is this?")
    uploaded_file2 = st.file_uploader(
        "Upload an image of Cauliflower, Plum, Corn or Ginger to detect the variation",
        type=["jpg", "png", "jpeg"],
        key="variation_upload",
    )

    if uploaded_file2:
        img2 = Image.open(uploaded_file2).convert("RGB")
        st.image(img2, caption="Your Image", use_container_width=True)

        # Preprocess for produce type detection
        tensor_produce = preprocess_image(
            img2, target_size=produce_model.input_shape[1:3]
        )
        detected_produce, _ = predict(produce_model, tensor_produce, produce_labels)

        # Fetch the corresponding variation model and labels
        var_model = variation_models[detected_produce]
        var_labels = variation_labels_map[detected_produce]

        # Preprocess again for the selected variation model (may have different input size)
        tensor_variation = preprocess_image(
            img2, target_size=var_model.input_shape[1:3]
        )
        variation_pred, variation_conf = predict(
            var_model, tensor_variation, var_labels
        )

        st.info(f"Detected produce: **{detected_produce}**")
        st.success(
            f"🧠 Predicted variation: **{variation_pred}** with **{variation_conf:.2f}%** confidence."
        )

# -------------------------
# Page 3: Scientific Report
# -------------------------
else:  # selection == "Scientific Report"
    st.subheader("📄 Scientific Report")
    
    # PDF paths
    pdf_path_jacky = os.path.join(
        os.path.dirname(__file__), "..", "Reports", "CV_Jhe48_Report.pdf"
    )
    pdf_path_mthameem = os.path.join(
        os.path.dirname(__file__), "..", "Reports", "cv_mthameem_report.pdf"
    )

    # Check if both PDFs exist
    if os.path.exists(pdf_path_jacky) and os.path.exists(pdf_path_mthameem):
        # Load both PDFs
        with open(pdf_path_jacky, "rb") as f:
            pdf_bytes_jacky = f.read()
        with open(pdf_path_mthameem, "rb") as f:
            pdf_bytes_mthameem = f.read()

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download Jacky He's Report",
                data=pdf_bytes_jacky,
                file_name="CV_Jhe48_Report.pdf",
                mime="application/pdf",
            )
        with col_dl2:
            st.download_button(
                label="Download Muhammad's Report",
                data=pdf_bytes_mthameem,
                file_name="cv_mthameem_report.pdf",
                mime="application/pdf",
            )

        # Convert to base64
        base64_pdf_jacky = base64.b64encode(pdf_bytes_jacky).decode("utf-8")
        base64_pdf_mthameem = base64.b64encode(pdf_bytes_mthameem).decode("utf-8")

        col1, col2 = st.columns(2, gap="large")

        # Left column – Jacky He
        with col1:
            st.markdown("### Jacky<br/>He", unsafe_allow_html=True)
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf_jacky}#view=FitH" '
                'style="width:100%; height:950px; border:none; overflow:hidden;" '
                'type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )

        # Right column – Muhammad Waseem Thameem Ansari
        with col2:
            st.markdown("### Muhammad Waseem Thameem Ansari")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf_mthameem}#view=FitH" '
                'style="width:100%; height:950px; border:none; overflow:hidden;" '
                'type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )
    else:
        st.error("One or both scientific reports not found.")
