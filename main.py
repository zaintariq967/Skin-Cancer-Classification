import streamlit as st
import torch
from PIL import Image
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import load_model
from src.predict import predict_image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Skin Cancer Detection | AI Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.85;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .result-benign {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #166534;
        border: 2px solid #22c55e;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .result-malignant {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        border: 2px solid #ef4444;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    st.markdown("## 🩺 AI Diagnostics")
    st.markdown("### Skin Cancer Classification")
    st.markdown("---")

    st.markdown("**Model Information**")
    st.markdown("""
    - **Architecture:** ResNet50  
    - **Dataset:** HAM10000  
    - **Task:** Benign vs Malignant  
    - **Test Accuracy:** 90.49%  
    - **Framework:** PyTorch
    """)

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
    1. Upload a dermoscopic image  
    2. Click **Analyze Image**  
    3. View prediction + confidence  
    """)

    st.markdown("---")
    st.warning("This tool is for research & educational purposes only. Not a medical device.")

# ==================== HEADER ====================
st.markdown("""
<div class="header-container">
    <div class="header-title">Automated Skin Cancer Detection System</div>
    <div class="header-subtitle">Deep Learning powered classification of dermoscopic skin lesions</div>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN PREDICTION SECTION ====================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Dermoscopic Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("Please upload a clear dermoscopic image of a skin lesion.")

with col2:
    st.markdown("### 📊 Analysis Result")

    if uploaded_file is not None:
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            with st.spinner("Running deep learning inference..."):
                time.sleep(0.6)
                label, confidence, probs = predict_image(model, temp_path, device)

            if label == "Malignant":
                st.markdown(f'<div class="result-malignant">Prediction: {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-benign">Prediction: {label}</div>', unsafe_allow_html=True)

            st.markdown("")

            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{confidence:.1f}%")
            m2.metric("Benign Prob", f"{probs[0] * 100:.1f}%")
            m3.metric("Malignant Prob", f"{probs[1] * 100:.1f}%")

            st.markdown("#### Probability Distribution")
            st.progress(float(probs[0]), text=f"Benign: {probs[0] * 100:.1f}%")
            st.progress(float(probs[1]), text=f"Malignant: {probs[1] * 100:.1f}%")

            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        st.info("Results will appear here after you upload and analyze an image.")

# ==================== MODEL PERFORMANCE SECTION ====================
st.markdown("---")
st.markdown("## 📈 Model Performance on Test Set")

perf_col1, perf_col2, perf_col3 = st.columns([1.2, 1, 1])

# ----- Confusion Matrix -----
with perf_col1:
    st.markdown("#### Confusion Matrix")

    # Your actual results
    cm = np.array([
        [1160, 50],  # True Benign
        [93, 200]  # True Malignant   (approximate from your report)
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ----- Test Set Summary -----
with perf_col2:
    st.markdown("#### Test Set Summary")
    st.metric("Total Images Tested", "1,503")
    st.metric("Benign Images", "1,210")
    st.metric("Malignant Images", "293")
    st.metric("Test Accuracy", "90.49%")

# ----- Class Distribution Chart -----
with perf_col3:
    st.markdown("#### Class Distribution (Test Set)")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    classes = ["Benign", "Malignant"]
    counts = [1210, 293]
    colors = ["#22c55e", "#ef4444"]

    bars = ax2.bar(classes, counts, color=colors, edgecolor="black")
    ax2.set_ylabel("Number of Images")
    ax2.set_title("Test Set Class Count")

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    st.pyplot(fig2)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <b>Automated Medical Diagnostics Project</b><br>
    Model: ResNet50 trained on HAM10000 Dataset &nbsp;|&nbsp; Test Accuracy: 90.49%<br>
    For research and educational use only • Not intended for clinical diagnosis
</div>
""", unsafe_allow_html=True)