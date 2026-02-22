# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# # ---------------- CUSTOM DESIGN ----------------
# st.markdown("""
#     <style>
#     .main-title {
#         font-size:28px !important;
#         font-weight:600;
#         text-align:center;
#         margin-bottom:10px;
#     }
#     .subtitle {
#         font-size:15px !important;
#         text-align:center;
#         color:gray;
#         margin-bottom:25px;
#     }
#     .result-box {
#         padding:15px;
#         border-radius:10px;
#         background-color:#f5f7fa;
#         text-align:center;
#         font-size:18px;
#         font-weight:500;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="main-title">🧠 MRI Brain Tumor Detection System</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Upload an MRI image to analyze tumor presence and type</div>', unsafe_allow_html=True)

# # ---------------- LOAD MODEL ----------------
# @st.cache_resource
# def load_trained_model():
#     return tf.keras.models.load_model("brain_tumor_detection_model.keras")

# model = load_trained_model()

# # 🔴 Replace with your real class_indices from Kaggle
# generator_class_indices = {
#     'glioma': 0,
#     'meningioma': 1,
#     'notumor': 2,
#     'pituitary': 3
# }

# sorted_labels = sorted(generator_class_indices.items(), key=lambda x: x[1])
# class_labels = [label for label, index in sorted_labels]

# # ---------------- FILE UPLOADER ----------------
# uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     # Smaller image display
#     st.image(image, caption="Uploaded MRI", width=250)

#     # -------- PREPROCESS --------
#     img = image.resize((224, 224))
#     img_array = np.array(img).astype("float32") / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # -------- PREDICTION --------
#     preds = model.predict(img_array)
#     pred_index = np.argmax(preds, axis=1)[0]
#     confidence = np.max(preds, axis=1)[0]

#     predicted_label = class_labels[pred_index]

#     if predicted_label.lower() == "notumor":
#         result = "✅ No Tumor Detected"
#         color = "green"
#     else:
#         result = f"⚠ Tumor Type: {predicted_label.capitalize()}"
#         color = "red"

#     # -------- RESULT DISPLAY --------
#     st.markdown(
#         f"""
#         <div class="result-box">
#             <span style="color:{color};">{result}</span><br><br>
#             Confidence: {confidence * 100:.2f}%
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import cv2  # for saturation check

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
    <style>
    .main-title {
        font-size:30px !important;
        font-weight:700;
        text-align:center;
        margin-bottom:15px;
        color:#1f2937;
    }
    .subtitle {
        font-size:16px !important;
        text-align:center;
        color:gray;
        margin-bottom:25px;
    }
    .result-box {
        padding:20px;
        border-radius:12px;
        background-color:#e0f2fe;
        text-align:center;
        font-size:18px;
        font-weight:600;
        margin-bottom:15px;
    }
    .tumor {
        color:red;
        font-weight:700;
    }
    .notumor {
        color:green;
        font-weight:700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 MRI Brain Tumor Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload one or two MRI images to analyze tumor presence and type</div>', unsafe_allow_html=True)

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
uploaded_files = st.file_uploader(
    "📤 Upload MRI Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### MRI Image {idx+1}")

        # --------------- Load & Validate Image ---------------
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Cannot open the uploaded file. Please upload a valid MRI image.")
            continue
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            continue

        # ---------------- MRI Input Check ----------------
        img_np = np.array(image)
        if img_np.ndim == 3:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            saturation_mean = hsv[:, :, 1].mean()
        else:
            saturation_mean = 0

        # If too colorful or bright → not brain MRI
        if saturation_mean > 60:
            st.error("❌ The uploaded image does not seem to be a brain MRI. Please upload a proper brain MRI scan.")
            continue

        # Smaller image display
        st.image(image, caption=f"Uploaded MRI {idx+1}", width=250)

        # --------------- Preprocess Image ---------------
        img = image.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --------------- Prediction ---------------
        with st.spinner("Analyzing MRI..."):
            try:
                preds = model.predict(img_array)
                pred_index = np.argmax(preds, axis=1)[0]
                confidence = np.max(preds, axis=1)[0]
            except Exception as e:
                st.error("❌ Prediction failed. Please check the uploaded image.")
                continue

        predicted_label = class_labels[pred_index]

        # --------------- Confidence Check ---------------
        if confidence < 0.5:
            st.warning("⚠ Low confidence in prediction. Consider uploading a clearer MRI scan.")

        # --------------- Result Display ---------------
        if predicted_label.lower() == "notumor":
            result_text = "✅ No Tumor Detected"
            css_class = "notumor"
        else:
            result_text = f"⚠ Tumor Type: {predicted_label.capitalize()}"
            css_class = "tumor"

        st.markdown(
            f"""
            <div class="result-box">
                <span class="{css_class}">{result_text}</span><br><br>
                Confidence: {confidence * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # --------------- Download Result ---------------
        st.download_button(
            f"📥 Download Result MRI {idx+1}",
            data=f"Prediction: {predicted_label.capitalize()}\nConfidence: {confidence*100:.2f}%",
            file_name=f"brain_tumor_result_{idx+1}.txt"
        )
