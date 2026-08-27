import streamlit as st
import torch
from PIL import Image
import os
import time
from src.model import load_model
from src.predict import predict_image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLE ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .result-benign {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        border: 2px solid #22c55e;
    }
    .result-malignant {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        border: 2px solid #ef4444;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODEL ====================
@st.cache_resource
def get_model():
    model_path = "assets/models/best_resnet50_full.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    return model, device


model, device = get_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🩺 About the System")
    st.markdown("""
    This application uses a **ResNet50** deep learning model trained on the **HAM10000** dataset to classify dermoscopic images as **Benign** or **Malignant**.

    **Performance:**
    - Test Accuracy: **90.49%**
    - Dataset: HAM10000
    - Framework: PyTorch
    """)
    st.markdown("---")
    st.info(
        "This tool is for **research and educational purposes only**. It is not a substitute for professional medical advice.")

# ==================== HEADER ====================
st.markdown('<p class="main-title">🩺 Automated Skin Cancer Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Benign vs Malignant Detection using Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== FILE UPLOADER ====================
uploaded_file = st.file_uploader("Upload a Dermoscopic Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Predict button
    if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
        with st.spinner("Analyzing image... Please wait"):
            time.sleep(0.7)
            label, confidence, probs = predict_image(model, temp_path, device)

        # ========== RESULT SECTION ==========
        st.markdown("---")
        st.subheader("Prediction Result")

        if label == "Malignant":
            st.markdown(f'<div class="result-malignant">Prediction: {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-benign">Prediction: {label}</div>', unsafe_allow_html=True)

        st.markdown("")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{confidence:.1f}%")

        if confidence >= 95:
            certainty = "Very High"
        elif confidence >= 80:
            certainty = "High"
        else:
            certainty = "Moderate"
        col2.metric("Certainty", certainty)
        col3.metric("Predicted Class", label)

        # Probability bars
        st.markdown("### Class Probabilities")
        p1, p2 = st.columns(2)
        with p1:
            st.write("**Benign**")
            st.progress(float(probs[0]))
            st.caption(f"{probs[0] * 100:.1f}%")
        with p2:
            st.write("**Malignant**")
            st.progress(float(probs[1]))
            st.caption(f"{probs[1] * 100:.1f}%")

        if confidence > 97:
            st.caption(
                "Note: The model is highly confident on this sample. Always consult a dermatologist for final diagnosis.")

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("Please upload a dermoscopic skin lesion image to get started.")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    Automated Medical Diagnostics Project &nbsp;|&nbsp; Model: ResNet50 (HAM10000) &nbsp;|&nbsp; Test Accuracy: 90.49%<br>
    For research and educational purposes only
</div>
""", unsafe_allow_html=True)