# Automated Skin Cancer Classification

### Deep Learning–Based Dermoscopic Image Classification using HAM10000

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green)](https://www.isic-archive.com/)
[![License](https://img.shields.io/badge/License-Academic%2FResearch-lightgrey)](#dataset)

> An end-to-end deep learning system for classifying dermoscopic skin-lesion images as **Benign** or **Malignant**, trained on the publicly available **HAM10000** dataset and deployed as an interactive Streamlit web application.

### Live Demo

**Try the deployed application:**

 [https://skin-cancer-classification-zain.streamlit.app/](https://skin-cancer-classification-zain.streamlit.app/)

---

## Project Overview

Skin-lesion classification is a challenging computer-vision problem because visually similar lesions can belong to very different diagnostic categories. Expert dermatological assessment is important but can be time-consuming and dependent on specialist availability.

This project explores the use of **Convolutional Neural Networks and transfer learning** to develop an accessible computer-aided classification system capable of analyzing dermoscopic images and producing a **Benign/Malignant prediction with confidence information**.

The system was developed as an **academic research and clinical decision-support prototype**. It is **not intended to replace professional medical diagnosis**. Final diagnostic decisions must always be made by qualified healthcare professionals.

The project requirements define an end-to-end workflow covering data preparation, preprocessing, model training, evaluation, model loading, and user-facing inference. 

---

## Objectives

The project has three primary objectives:

1. **Develop a deep learning classifier** using CNNs and transfer learning for dermoscopic skin-lesion classification.
2. **Build an end-to-end application** integrating preprocessing, model inference, evaluation, and a user-friendly interface.
3. **Evaluate the model rigorously** using Accuracy, Precision, Recall, F1-Score, AUC-ROC, confusion matrix, and ROC analysis. 

---

## Machine Learning Approach

The original HAM10000 dataset contains **7 lesion categories**. For this project, the problem was converted into a **binary classification task**:

| Class            | Description                               |
| ---------------- | ----------------------------------------- |
| **Benign**    | Non-malignant skin lesions                |
| **Malignant** | Malignant / higher-risk lesion categories |

The project groups the original HAM10000 classes as follows:

### Benign

* Melanocytic nevi (`nv`)
* Benign keratosis (`bkl`)
* Vascular lesions (`vasc`)
* Dermatofibroma (`df`)

### Malignant

* Melanoma (`mel`)
* Basal cell carcinoma (`bcc`)
* Actinic keratosis (`akiec`)

This binary grouping is defined in the project SRS. 

---

# Dataset

The project uses the **HAM10000 — Human Against Machine with 10000 training images** dataset.

| Property                 | Value                          |
| ------------------------ | ------------------------------ |
| Dataset                  | HAM10000                       |
| Total Images             | **10,015**                     |
| Image Type               | Dermoscopic                    |
| Original Classes         | **7**                          |
| Approx. Image Resolution | **600 × 450**                  |
| Source                   | ISIC Archive / Scientific Data |
| Task                     | Binary Classification          |

The original class distribution is highly imbalanced, with melanocytic nevi representing the largest class. 

### Original HAM10000 Classes

| Code    | Lesion               |
| ------- | -------------------- |
| `nv`    | Melanocytic nevi     |
| `mel`   | Melanoma             |
| `bkl`   | Benign keratosis     |
| `bcc`   | Basal cell carcinoma |
| `akiec` | Actinic keratosis    |
| `vasc`  | Vascular lesions     |
| `df`    | Dermatofibroma       |

### Dataset Split

A stratified split was used:

* **70% — Training**
* **15% — Validation**
* **15% — Testing**

This ensures that class proportions are maintained across the different subsets. 

---

# Image Preprocessing

All images are processed consistently before being passed to the neural network.

### Training Pipeline

```text
Original Image
      ↓
Resize to 224 × 224
      ↓
Random Horizontal Flip
      ↓
Random Vertical Flip
      ↓
Random Rotation
      ↓
Color Jitter
      ↓
Convert to Tensor
      ↓
ImageNet Normalization
      ↓
Model
```

### Validation / Testing Pipeline

```text
Original Image
      ↓
Resize to 224 × 224
      ↓
Convert to Tensor
      ↓
ImageNet Normalization
      ↓
Model
```

Data augmentation is applied only during training to improve generalization.

---

# Model Development

Instead of selecting a single architecture immediately, multiple transfer-learning architectures were investigated.

### Models Compared

* **ResNet50**
* **EfficientNet-B0**
* **DenseNet121**

All models were adapted for the two-class classification problem.

### Model Selection

After comparison, **ResNet50** was selected as the final architecture based on validation performance.

The project architecture follows the general pipeline:

```text
HAM10000 Dataset
       │
       ▼
Data Preparation
       │
       ▼
Preprocessing & Augmentation
       │
       ▼
Transfer Learning
       │
       ├── ResNet50
       ├── EfficientNet-B0
       └── DenseNet121
              │
              ▼
       Model Comparison
              │
              ▼
          ResNet50
              │
              ▼
       Final Evaluation
              │
              ▼
      Streamlit Application
```

The SRS defines the system architecture around a data layer, preprocessing layer, CNN/model layer, inference layer, and presentation layer. 

---

# Training Configuration

The final training workflow used:

| Parameter      | Configuration      |
| -------------- | ------------------ |
| Architecture   | **ResNet50**       |
| Framework      | **PyTorch**        |
| Input Size     | **224 × 224**      |
| Batch Size     | **32**             |
| Optimizer      | **Adam**           |
| Learning Rate  | **1 × 10⁻⁴**       |
| Loss Function  | Cross Entropy Loss |
| LR Scheduler   | ReduceLROnPlateau  |
| Early Stopping | Enabled            |
| Classification | Binary             |
| Classes        | Benign / Malignant |

The final ResNet50 model was subsequently trained using the full available dataset split.

---

# Final Model Performance

The final **ResNet50** model achieved the following results on the held-out test set:

| Metric              |      Score |
| ------------------- | ---------: |
| **Test Accuracy**   | **90.49%** |
| Benign Precision    |   **0.93** |
| Benign Recall       |   **0.96** |
| Benign F1-Score     |   **0.94** |
| Malignant Precision |   **0.80** |
| Malignant Recall    |   **0.68** |
| Malignant F1-Score  |   **0.74** |
| Macro F1-Score      |   **0.84** |
| Weighted F1-Score   |   **0.90** |

### Test Set

* Total test images: **1,503**
* Benign: **1,210**
* Malignant: **293**

The notebook also evaluates the final model using:

* Classification Report
* Confusion Matrix
* ROC Curve
* AUC-ROC

The project SRS explicitly identifies these metrics as part of model validation. 

> **Important:** Overall accuracy should not be interpreted in isolation. The malignant-class recall reported by this experiment is lower than benign recall, which is an important limitation for a medical classification task.

---

# Streamlit Application

The trained model is integrated into an interactive **Streamlit** web application.

### Application Workflow

```text
Upload Dermoscopic Image
          ↓
Image Validation
          ↓
Preprocessing
          ↓
ResNet50 Model
          ↓
Prediction
          ↓
Benign / Malignant
          ↓
Confidence Score
```

### Main Capabilities

* Upload dermoscopic images
* Preview the uploaded image
* Automatic preprocessing
* Deep-learning inference
* Benign prediction
* Malignant prediction
* Confidence/probability information
* Interactive web-based interface
* Cloud deployment through Streamlit

The intended interface includes image upload, classification, result visualization, and confidence information. 

---

# Technology Stack

### Programming

* Python

### Deep Learning

* PyTorch
* Torchvision
* ResNet50
* Transfer Learning

### Machine Learning / Evaluation

* Scikit-learn
* Classification Report
* Confusion Matrix
* ROC-AUC

### Image Processing

* PIL / Pillow
* Torchvision Transforms

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

---

# Project Structure

A recommended repository structure is:

```text
Skin-Cancer-Classification/
│
├── assets/
│   ├── notebook/
|            └── Skin-Cancer-Classification_MZainTariq.ipynb
│   ├── srs/
|         └──  SkinCancer_MZainTariq.pdf
|   ├── model/
│        └── best_resnet50_full.pth
|         └── best_resnet50_full.pkl
|   ├── test_imgs/
│        
│
├── app.py
├── main.py 
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Dataset files are not included directly in the repository because of their size and dataset licensing considerations.

---

# Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

---

# Model

The final trained model is a **ResNet50** network trained for binary classification.

The notebook saves the model in PyTorch format:

```text
best_resnet50_full.pth
```

A `.pkl` version is also generated as part of the project workflow.

The SRS specifies trained model weights as one of the project's expected deliverables. 

---

# Evaluation

The model evaluation process includes:

### 1. Accuracy

Measures the overall percentage of correctly classified samples.

### 2. Precision

Measures how many samples predicted as a particular class were actually from that class.

### 3. Recall

Measures how many actual samples from a class were successfully detected.

### 4. F1-Score

Provides a balance between precision and recall.

### 5. Confusion Matrix

Provides a detailed view of correct and incorrect predictions for each class.

### 6. ROC-AUC

Evaluates the model's ability to distinguish between the two classes across classification thresholds.

These evaluation methods form part of the project's defined verification and validation strategy. 

---

# Medical Disclaimer

> **This application is an educational and research prototype and must NOT be used as a substitute for professional medical diagnosis.**

The model was trained on the HAM10000 dataset and may not generalize to every population, imaging device, clinical environment, skin type, or real-world lesion.

A prediction from this application should **not** be considered a medical diagnosis or treatment recommendation.

If you have a concerning or changing skin lesion, consult a qualified dermatologist or healthcare professional.

The project's SRS explicitly defines the system as a clinical decision-support/research prototype rather than a definitive diagnostic system. 

---

# Privacy & Security

The system is designed without requiring patient-identifiable information.

According to the project requirements:

* No patient-identifiable information is required.
* Uploaded images are processed for inference.
* Patient data is not intended to be permanently stored.
* The core classification pipeline does not depend on external paid APIs.



---

# Dataset & References

### HAM10000

Tschandl, P., Rosendahl, C., & Kittler, H. (2018).

**The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.**

Scientific Data, 5, 180161.

Dataset / archive:

[https://www.isic-archive.com/](https://www.isic-archive.com/)

The project SRS identifies HAM10000 as the sole primary dataset containing 10,015 dermoscopic images across seven original classes. 

---

# Future Improvements

Potential future development includes:

* Explainable AI using **Grad-CAM**
* Mobile application
* Integration with hospital information systems
* REST API
* Multi-disease classification
* Improved class balancing
* Improved malignant-class recall
* Larger and more diverse clinical datasets
* More extensive external validation
* Better uncertainty estimation and calibration

Grad-CAM, mobile deployment, hospital-system integration, and broader disease classification are identified as future enhancements in the project's SRS. 

---

# Author

### Engr. Muhammad Zain Tariq

**Computer Engineering | Artificial Intelligence | Machine Learning | Deep Learning**

Pakistan

---

# Project Highlights

```text
┌────────────────────────────────────────────────────┐
│          SKIN CANCER CLASSIFICATION                │
├────────────────────────────────────────────────────┤
│                                                    │
│  Dataset       → HAM10000                         │
│  Images        → 10,015                           │
│  Task          → Binary Classification             │
│  Models        → ResNet50 / EfficientNet / DenseNet│
│  Final Model   → ResNet50                         │
│  Framework     → PyTorch                           │
│  Test Accuracy → 90.49%                           │
│  Deployment    → Streamlit                         │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Live Application

### **[Launch Skin Cancer Classification App](https://skin-cancer-classification-zain.streamlit.app/)**

Upload a dermoscopic image and explore the model's prediction workflow through the deployed Streamlit interface.

---

## Documentation

The complete Software Requirements Specification covers:

* Problem statement
* System scope
* Functional requirements
* Non-functional requirements
* System architecture
* Dataset requirements
* Model objectives
* Interface requirements
* Verification and validation
* Future enhancements

The SRS was prepared as part of the project documentation. 

---

## If You Find This Project Useful

If this project is useful for your learning or research:

**Star the repository**

**Fork the repository**
0
**Open an issue or discussion for suggestions**

---

### Project Status

**Model Development:** Completed
**Model Evaluation:** Completed
**Streamlit Application:** Deployed
**Documentation:** ✅ Available
**Research Prototype:** ✅ Completed
