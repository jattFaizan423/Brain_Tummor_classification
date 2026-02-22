import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
    <style>
    .main-title {
        font-size:28px !important;
        font-weight:600;
        text-align:center;
        margin-bottom:10px;
    }
    .subtitle {
        font-size:15px !important;
        text-align:center;
        color:gray;
        margin-bottom:25px;
    }
    .result-box {
        padding:15px;
        border-radius:10px;
        background-color:#f5f7fa;
        text-align:center;
        font-size:18px;
        font-weight:500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 MRI Brain Tumor Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an MRI image to analyze tumor presence and type</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("brain_tumor_detection_model.keras")

model = load_trained_model()

# 🔴 Replace with your real class_indices from Kaggle
generator_class_indices = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

sorted_labels = sorted(generator_class_indices.items(), key=lambda x: x[1])
class_labels = [label for label, index in sorted_labels]

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Smaller image display
    st.image(image, caption="Uploaded MRI", width=250)

    # -------- PREPROCESS --------
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- PREDICTION --------
    preds = model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds, axis=1)[0]

    predicted_label = class_labels[pred_index]

    if predicted_label.lower() == "notumor":
        result = "✅ No Tumor Detected"
        color = "green"
    else:
        result = f"⚠ Tumor Type: {predicted_label.capitalize()}"
        color = "red"

    # -------- RESULT DISPLAY --------
    st.markdown(
        f"""
        <div class="result-box">
            <span style="color:{color};">{result}</span><br><br>
            Confidence: {confidence * 100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )


