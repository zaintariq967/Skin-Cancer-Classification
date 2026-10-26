"""
evaluate.py  -  Skin Cancer Classification  |  Model Evaluation Script
=======================================================================
ResNet50  |  HAM10000 dataset  |  Benign vs Malignant binary classification

Works with a FLAT image folder + HAM10000_metadata.csv for true labels.
No benign/malignant subfolders needed.

Outputs saved to  outputs/  folder:
  1.  confusion_matrix.png          - Raw counts + Normalised (%) side by side
  2.  roc_curve.png                 - ROC with AUC score
  3.  precision_recall_curve.png    - PR curve + Average Precision (AP)
  4.  probability_histogram.png     - P(Malignant) distributions per class
  5.  per_class_accuracy.png        - Per-class Accuracy + Precision/Recall/F1
  6.  metrics_heatmap.png           - Correlation-style heatmap of all metrics
  7.  gradcam_samples.png           - Grad-CAM heatmaps on sample images
  8.  misclassified_grid.png        - Worst false positives & false negatives
  9.  summary_dashboard.png         - Single-page 2x3 overview dashboard
  10. classification_report.txt     - Full sklearn text report
  11. metrics_summary.json          - Structured JSON of all metrics
  12. predictions_table.csv         - Per-image predictions

Usage (zero arguments, uses defaults):
  python evaluate.py

With custom paths:
  python evaluate.py
      --img_dir     assets/test_imgs
      --metadata    path/to/HAM10000_metadata.csv
      --model_path  assets/models/best_resnet50_full.pth
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES       = ["Benign", "Malignant"]
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}   # same mapping used in training
OUTPUT_DIR        = "outputs"
BATCH_SIZE        = 32
IMAGE_SIZE        = 224
GRADCAM_SAMPLES   = 8   # number of sample images for Grad-CAM

PAL = {"Benign": "#22c55e", "Malignant": "#ef4444", "accent": "#3b82f6"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory : {os.path.abspath(OUTPUT_DIR)}\n")


def save_fig(fig, name: str, dpi: int = 150):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME  (flat folder + HAM10000 metadata CSV)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(img_dir: str, metadata_path) -> pd.DataFrame:
    """
    Match images in img_dir to their true labels via HAM10000_metadata.csv.
    Falls back to prediction-only mode if no CSV is given.
    """
    exts = {".jpg", ".jpeg", ".png"}
    img_files = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not img_files:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    print(f"[OK] Found {len(img_files)} images in: {img_dir}")

    records = []
    for fname in img_files:
        image_id = os.path.splitext(fname)[0]
        records.append({
            "image_id": image_id,
            "filename": fname,
            "filepath": os.path.join(img_dir, fname),
        })
    df = pd.DataFrame(records)

    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        print(f"[OK] Metadata loaded : {len(meta)} rows  |  {metadata_path}")

        meta["label"] = meta["dx"].apply(
            lambda x: 1 if str(x).lower() in MALIGNANT_CLASSES else 0
        )
        meta["label_name"] = meta["label"].map({0: "Benign", 1: "Malignant"})

        df = df.merge(meta[["image_id", "dx", "label", "label_name"]],
                      on="image_id", how="left")

        missing = df["label"].isna().sum()
        if missing:
            print(f"[!] {missing} image(s) not found in metadata — skipping.")

        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        print(f"[OK] Matched  {len(df)} images with labels.")
        print(f"     Benign : {(df['label']==0).sum()}  |  "
              f"Malignant : {(df['label']==1).sum()}")
    else:
        print("[!] No metadata CSV — PREDICTION-ONLY mode (no ground-truth metrics).")
        df["label"]      = -1
        df["label_name"] = "Unknown"
        df["dx"]         = "Unknown"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FlatImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = INFER_TRANSFORM(image)
        label = int(row["label"]) if row["label"] != -1 else -1
        return image, label, row["image_id"], row["filepath"]


def collate_fn(batch):
    images    = torch.stack([b[0] for b in batch])
    labels    = [b[1] for b in batch]
    ids       = [b[2] for b in batch]
    filepaths = [b[3] for b in batch]
    return images, labels, ids, filepaths


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded    : {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, df: pd.DataFrame, device: str):
    dataset = FlatImageDataset(df)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    all_ids, all_fps, all_labels, all_preds, all_probs = [], [], [], [], []

    print("\nRunning inference ...")
    with torch.no_grad():
        for images, labels, ids, fps in tqdm(loader, unit="batch"):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_ids.extend(ids)
            all_fps.extend(fps)
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_ids),
        np.array(all_fps),
        np.array(all_labels, dtype=int),
        np.array(all_preds,  dtype=int),
        np.array(all_probs),         # shape [N, 2]
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM for ResNet50: hooks onto layer4."""

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Hook on final conv block
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a [H, W] heatmap normalised to [0, 1]."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # [h, w]
        cam     = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()


def overlay_heatmap(original_pil: Image.Image, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap over original image."""
    heatmap  = Image.fromarray(np.uint8(255 * cam)).resize(
        original_pil.size, Image.BILINEAR
    )
    heatmap_rgb = np.array(
        plt.cm.jet(np.array(heatmap) / 255.0)[:, :, :3] * 255,
        dtype=np.uint8
    )
    original_np = np.array(original_pil)
    overlay     = (alpha * heatmap_rgb + (1 - alpha) * original_np).astype(np.uint8)
    return overlay


def plot_gradcam_samples(model, df: pd.DataFrame, y_pred: np.ndarray,
                         y_probs: np.ndarray, device: str):
    """
    Plot Grad-CAM for a sample of images (mix of benign & malignant predictions).
    """
    gradcam = GradCAM(model)
    n       = min(GRADCAM_SAMPLES, len(df))
    indices = np.linspace(0, len(df) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(14, n * 3.2))
    fig.suptitle("Grad-CAM Visualizations  |  ResNet50 (layer4)",
                 fontsize=14, fontweight="bold", y=1.01)

    if n == 1:
        axes = [axes]

    for row_idx, img_idx in enumerate(indices):
        filepath  = df.iloc[img_idx]["filepath"]
        image_id  = df.iloc[img_idx]["image_id"]
        true_lbl  = df.iloc[img_idx]["label"]
        pred_lbl  = int(y_pred[img_idx])
        conf      = float(y_probs[img_idx][pred_lbl]) * 100

        # Load original image
        original = Image.open(filepath).convert("RGB")

        # Prepare tensor
        input_t = INFER_TRANSFORM(original).unsqueeze(0).to(device)
        input_t.requires_grad_(False)

        # Enable grad for Grad-CAM
        model.train()   # activate batch norm tracking
        model.eval()
        with torch.enable_grad():
            cam = gradcam.generate(input_t, pred_lbl)

        overlay = overlay_heatmap(original, cam)

        # Column 0: original
        axes[row_idx][0].imshow(original)
        axes[row_idx][0].set_title(f"{image_id}\nOriginal", fontsize=8)
        axes[row_idx][0].axis("off")

        # Column 1: heatmap only
        axes[row_idx][1].imshow(cam, cmap="jet")
        axes[row_idx][1].set_title("Grad-CAM", fontsize=8)
        axes[row_idx][1].axis("off")

        # Column 2: overlay
        true_str = CLASS_NAMES[true_lbl] if true_lbl != -1 else "?"
        color    = "#22c55e" if pred_lbl == 0 else "#ef4444"
        axes[row_idx][2].imshow(overlay)
        axes[row_idx][2].set_title(
            f"Pred: {CLASS_NAMES[pred_lbl]} ({conf:.1f}%)\nTrue: {true_str}",
            fontsize=8, color=color, fontweight="bold"
        )
        axes[row_idx][2].axis("off")

    plt.tight_layout()
    return save_fig(fig, "gradcam_samples.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold", y=1.02)

    # Raw counts
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"}, ax=axes[0])
    axes[0].set_title("Raw Counts", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    # Normalised (%)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"}, ax=axes[1])
    axes[1].set_title("Normalised (%)", fontsize=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    return save_fig(fig, "confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PAL["accent"], lw=2.5,
            label=f"ResNet50  AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PAL["accent"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.50)")

    # Mark operating point at threshold=0.5
    thresh_idx = np.argmin(np.abs(fpr + (1 - tpr) - 1))
    ax.scatter(fpr[thresh_idx], tpr[thresh_idx], color="red", zorder=5,
               s=80, label=f"Threshold=0.5  (FPR={fpr[thresh_idx]:.2f}, TPR={tpr[thresh_idx]:.2f})")

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return save_fig(fig, "roc_curve.png"), roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – PRECISION-RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curve(y_true, y_probs):
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs[:, 1])
    ap = average_precision_score(y_true, y_probs[:, 1])

    # Baseline (random)
    baseline = y_true.sum() / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(rec_arr, prec_arr, where="post", color=PAL["Malignant"], lw=2.5,
            label=f"ResNet50  AP = {ap:.4f}")
    ax.fill_between(rec_arr, prec_arr, alpha=0.12, color=PAL["Malignant"], step="post")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1.5,
               label=f"Random (AP = {baseline:.2f})")

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve  |  Malignant class",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return save_fig(fig, "precision_recall_curve.png"), ap


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – METRICS HEATMAP  (correlation-style table)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(y_true, y_pred, y_probs, roc_auc, ap):
    """
    Heatmap showing all key metrics per class and overall,
    similar to a correlation matrix but for classification metrics.
    """
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    overall_acc = accuracy_score(y_true, y_pred) * 100
    mcc         = matthews_corrcoef(y_true, y_pred)
    kappa       = cohen_kappa_score(y_true, y_pred)

    # Build metrics table  [metrics x classes]
    metrics_names  = ["Precision (%)", "Recall (%)", "F1-Score (%)",
                       "Specificity (%)", "Support"]
    cm_arr         = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_arr.ravel()

    specificity_benign    = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    specificity_malignant = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    data = np.array([
        [prec[0] * 100, prec[1] * 100],
        [rec[0]  * 100, rec[1]  * 100],
        [f1[0]   * 100, f1[1]   * 100],
        [specificity_benign, specificity_malignant],
        [float(support[0]),  float(support[1])],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Metrics Heatmap  |  ResNet50  |  HAM10000",
                 fontsize=14, fontweight="bold")

    # Left: per-class metrics heatmap
    df_heat = pd.DataFrame(data, index=metrics_names, columns=CLASS_NAMES)
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"},
                vmin=0, vmax=100, ax=axes[0])
    axes[0].set_title("Per-Class Metrics", fontsize=12)
    axes[0].set_xlabel("Class", fontsize=11)
    axes[0].set_ylabel("Metric", fontsize=11)

    # Right: overall summary as a small heatmap
    overall_data = np.array([
        [overall_acc],
        [roc_auc * 100],
        [ap * 100],
        [mcc * 100],
        [kappa * 100],
    ])
    overall_labels = ["Accuracy (%)", "ROC AUC (%)", "Avg Precision (%)",
                      "MCC (%)", "Cohen Kappa (%)"]
    df_overall = pd.DataFrame(overall_data, index=overall_labels,
                               columns=["Overall"])
    sns.heatmap(df_overall, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"},
                vmin=0, vmax=100, ax=axes[1])
    axes[1].set_title("Overall Metrics", fontsize=12)
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    plt.tight_layout()
    return save_fig(fig, "metrics_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – PROBABILITY HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_probability_histogram(y_true, y_probs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Predicted Probability Distributions  |  P(Malignant)",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, (cls_name, color) in enumerate(
        zip(CLASS_NAMES, [PAL["Benign"], PAL["Malignant"]])
    ):
        probs_cls = y_probs[y_true == idx, 1]
        axes[idx].hist(probs_cls, bins=40, color=color, edgecolor="white", alpha=0.85)
        axes[idx].axvline(0.5, color="black", linestyle="--", lw=1.5,
                          label="Threshold = 0.5")
        mean_p = probs_cls.mean()
        axes[idx].axvline(mean_p, color="navy", linestyle="-.", lw=1.5,
                          label=f"Mean = {mean_p:.2f}")
        axes[idx].set_title(f"True Class: {cls_name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("P(Malignant)", fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "probability_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 – PER-CLASS ACCURACY + PRECISION/RECALL/F1
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_accuracy(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-Class Performance Metrics", fontsize=14, fontweight="bold")

    accs = []
    for c in range(len(CLASS_NAMES)):
        mask = y_true == c
        acc  = accuracy_score(y_true[mask], y_pred[mask]) * 100
        accs.append(acc)

    bars = axes[0].bar(CLASS_NAMES, accs,
                       color=[PAL["Benign"], PAL["Malignant"]],
                       edgecolor="white", width=0.45)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title("Per-Class Accuracy", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    colors = ["#6366f1", "#f59e0b", "#10b981"]
    for i, (values, metric) in enumerate(
        zip([prec * 100, rec * 100, f1 * 100], ["Precision", "Recall", "F1-Score"])
    ):
        rects = axes[1].bar(x + i * w, values, w, label=metric,
                            color=colors[i], edgecolor="white")
        for rect, val in zip(rects, values):
            axes[1].text(rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + 0.5,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x + w); axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylim(0, 115); axes[1].set_ylabel("Score (%)", fontsize=11)
    axes[1].set_title("Precision / Recall / F1", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "per_class_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 – MISCLASSIFIED IMAGES GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_misclassified_grid(df: pd.DataFrame, y_true, y_pred, y_probs,
                            n_show: int = 8):
    """Show worst false positives (Benign predicted Malignant) and
    worst false negatives (Malignant predicted Benign) sorted by confidence."""
    has_labels = (y_true != -1).all()
    if not has_labels:
        print("[!] Skipping misclassified grid — no ground-truth labels.")
        return

    indices = np.arange(len(y_true))

    # False positives: true=0, pred=1 (sorted by P(Malignant) descending)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_idxs = indices[fp_mask]
    fp_idxs = fp_idxs[np.argsort(-y_probs[fp_idxs, 1])][:n_show // 2]

    # False negatives: true=1, pred=0 (sorted by P(Benign) descending)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_idxs = indices[fn_mask]
    fn_idxs = fn_idxs[np.argsort(-y_probs[fn_idxs, 0])][:n_show // 2]

    all_idxs = list(fp_idxs) + list(fn_idxs)
    if not all_idxs:
        print("[!] No misclassified samples found — skipping grid.")
        return

    ncols = 4
    nrows = (len(all_idxs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Misclassified Samples  |  FP (red border) & FN (orange border)",
                 fontsize=13, fontweight="bold")

    for ax_i, img_idx in enumerate(all_idxs):
        filepath = df.iloc[img_idx]["filepath"]
        img      = Image.open(filepath).convert("RGB")
        true_lbl = int(y_true[img_idx])
        pred_lbl = int(y_pred[img_idx])
        conf     = float(y_probs[img_idx][pred_lbl]) * 100
        is_fp    = (true_lbl == 0 and pred_lbl == 1)

        axes[ax_i].imshow(img)
        axes[ax_i].set_title(
            f"True: {CLASS_NAMES[true_lbl]}\n"
            f"Pred: {CLASS_NAMES[pred_lbl]} ({conf:.1f}%)",
            fontsize=8,
            color="#ef4444" if is_fp else "#f59e0b",
            fontweight="bold",
        )
        border_color = "#ef4444" if is_fp else "#f59e0b"
        for spine in axes[ax_i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        axes[ax_i].set_xticks([]); axes[ax_i].set_yticks([])

    # Hide unused axes
    for ax_i in range(len(all_idxs), len(axes)):
        axes[ax_i].axis("off")

    plt.tight_layout()
    return save_fig(fig, "misclassified_grid.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 – SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(y_true, y_pred, y_probs, roc_auc, ap):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Model Evaluation Dashboard  |  ResNet50  |  HAM10000  |  Benign vs Malignant",
        fontsize=15, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.45)

    # 1. Confusion matrix
    ax1  = fig.add_subplot(gs[0, 0])
    cm_a = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax1)
    ax1.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=9); ax1.set_ylabel("Actual", fontsize=9)

    # 2. ROC
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    ax2.plot(fpr, tpr, color=PAL["accent"], lw=2, label=f"AUC = {roc_auc:.4f}")
    ax2.fill_between(fpr, tpr, alpha=0.08, color=PAL["accent"])
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("FPR", fontsize=9); ax2.set_ylabel("TPR", fontsize=9)
    ax2.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(linestyle="--", alpha=0.4)

    # 3. Precision-Recall
    ax3 = fig.add_subplot(gs[0, 2])
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs[:, 1])
    ax3.step(rec_arr, prec_arr, where="post", color=PAL["Malignant"], lw=2,
             label=f"AP = {ap:.4f}")
    ax3.fill_between(rec_arr, prec_arr, alpha=0.10,
                     color=PAL["Malignant"], step="post")
    baseline = y_true.sum() / len(y_true)
    ax3.axhline(baseline, color="k", linestyle="--", lw=1.2,
                label=f"Baseline = {baseline:.2f}")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Recall", fontsize=9); ax3.set_ylabel("Precision", fontsize=9)
    ax3.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(linestyle="--", alpha=0.4)

    # 4. Prob histogram - Benign
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(y_probs[y_true == 0, 1], bins=35, color=PAL["Benign"],
             edgecolor="white", alpha=0.85)
    ax4.axvline(0.5, color="black", linestyle="--", lw=1.5)
    ax4.set_title("P(Malignant) | True Benign", fontsize=11, fontweight="bold")
    ax4.set_xlabel("P(Malignant)", fontsize=9)
    ax4.set_ylabel("Count", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    # 5. Prob histogram - Malignant
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_probs[y_true == 1, 1], bins=35, color=PAL["Malignant"],
             edgecolor="white", alpha=0.85)
    ax5.axvline(0.5, color="black", linestyle="--", lw=1.5)
    ax5.set_title("P(Malignant) | True Malignant", fontsize=11, fontweight="bold")
    ax5.set_xlabel("P(Malignant)", fontsize=9)
    ax5.set_ylabel("Count", fontsize=9)
    ax5.grid(axis="y", linestyle="--", alpha=0.4)

    # 6. Precision / Recall / F1 bar
    ax6  = fig.add_subplot(gs[1, 2])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    x = np.arange(len(CLASS_NAMES)); w = 0.25
    ax6.bar(x - w, prec * 100, w, label="Precision", color="#6366f1", edgecolor="white")
    ax6.bar(x,     rec  * 100, w, label="Recall",    color="#f59e0b", edgecolor="white")
    ax6.bar(x + w, f1   * 100, w, label="F1",        color="#10b981", edgecolor="white")
    ax6.set_xticks(x); ax6.set_xticklabels(CLASS_NAMES)
    ax6.set_ylim(0, 115); ax6.set_ylabel("Score (%)", fontsize=9)
    ax6.set_title("Precision / Recall / F1", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=8); ax6.grid(axis="y", linestyle="--", alpha=0.4)

    return save_fig(fig, "summary_dashboard.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION-ONLY PLOT (no labels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_only(y_pred, y_probs):
    benign_count    = int((y_pred == 0).sum())
    malignant_count = int((y_pred == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction Summary  |  ResNet50  |  HAM10000",
                 fontsize=13, fontweight="bold")

    axes[0].pie([benign_count, malignant_count],
                labels=CLASS_NAMES,
                colors=[PAL["Benign"], PAL["Malignant"]],
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Predicted Class Distribution", fontsize=12)

    axes[1].hist(y_probs[:, 1], bins=40, color=PAL["accent"],
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
    axes[1].set_xlabel("P(Malignant)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("P(Malignant) Distribution", fontsize=12)
    axes[1].legend(); axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "predictions_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT + JSON METRICS
# ─────────────────────────────────────────────────────────────────────────────

def save_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    path   = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(path, "w") as f:
        f.write("Skin Cancer Classification  |  ResNet50 / HAM10000\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
    print(f"[OK] Saved -> {path}")
    print("\n" + "=" * 55)
    print("Classification Report:")
    print("=" * 55)
    print(report)
    return path


def save_metrics_json(y_true, y_pred, y_probs, roc_auc, ap):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm_arr         = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_arr.ravel()
    overall_acc    = accuracy_score(y_true, y_pred) * 100
    mcc            = matthews_corrcoef(y_true, y_pred)
    kappa          = cohen_kappa_score(y_true, y_pred)
    specificity    = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    sensitivity    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    metrics = {
        "model":              "ResNet50",
        "dataset":            "HAM10000",
        "task":               "Benign vs Malignant",
        "overall_accuracy_%": round(overall_acc, 4),
        "roc_auc":            round(float(roc_auc), 4),
        "avg_precision_ap":   round(float(ap), 4),
        "mcc":                round(float(mcc), 4),
        "cohen_kappa":        round(float(kappa), 4),
        "specificity_%":      round(float(specificity), 4),
        "sensitivity_%":      round(float(sensitivity), 4),
        "total_samples":      int(len(y_true)),
        "per_class": {
            cls: {
                "precision_%": round(float(prec[i]) * 100, 4),
                "recall_%":    round(float(rec[i])  * 100, 4),
                "f1_%":        round(float(f1[i])   * 100, 4),
                "support":     int(support[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm_arr.tolist(),
    }
    path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved -> {path}")
    return metrics


def save_predictions_csv(image_ids, y_true, y_pred, y_probs):
    rows = []
    for img_id, true, pred, prob in zip(image_ids, y_true, y_pred, y_probs):
        rows.append({
            "image_id":         img_id,
            "true_label":       CLASS_NAMES[true] if true != -1 else "Unknown",
            "predicted_label":  CLASS_NAMES[pred],
            "prob_benign_%":    round(float(prob[0]) * 100, 2),
            "prob_malignant_%": round(float(prob[1]) * 100, 2),
            "correct":          (true == pred) if true != -1 else "N/A",
        })
    results_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "predictions_table.csv")
    results_df.to_csv(path, index=False)
    print(f"[OK] Saved -> {path}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation of the Skin Cancer ResNet50 model."
    )
    parser.add_argument("--img_dir",    type=str,
                        default="assets/test_imgs",
                        help="Flat folder of test images (default: assets/test_imgs).")
    parser.add_argument("--metadata",   type=str, default=None,
                        help="Path to HAM10000_metadata.csv. Omit for prediction-only mode.")
    parser.add_argument("--model_path", type=str,
                        default="assets/models/best_resnet50_full.pth",
                        help="Path to trained .pth model weights.")
    parser.add_argument("--no_gradcam", action="store_true",
                        help="Skip Grad-CAM (faster run, no extra memory).")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 65}")
    print("  Skin Cancer Classification  |  Model Evaluation")
    print(f"{'=' * 65}")
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model_path}")
    print(f"  Image dir   : {args.img_dir}")
    print(f"  Metadata    : {args.metadata or '(none — prediction-only mode)'}")
    print(f"  Output dir  : {OUTPUT_DIR}\n")

    make_output_dir()

    df    = build_dataframe(args.img_dir, args.metadata)
    model = load_model(args.model_path, device)

    image_ids, filepaths, y_true, y_pred, y_probs = run_inference(
        model, df, device
    )

    # Rebuild df to match filtered/sorted inference results
    id_to_row = {row["image_id"]: row for _, row in df.iterrows()}
    inf_df    = pd.DataFrame([id_to_row[iid] for iid in image_ids
                               if iid in id_to_row]).reset_index(drop=True)

    print(f"\n{'=' * 55}")
    print("  Saving outputs ...")
    print(f"{'=' * 55}")

    save_predictions_csv(image_ids, y_true, y_pred, y_probs)

    has_labels = (y_true != -1).all()

    if has_labels:
        plot_confusion_matrix(y_true, y_pred)
        _, roc_auc = plot_roc_curve(y_true, y_probs)
        _, ap      = plot_precision_recall_curve(y_true, y_probs)
        plot_metrics_heatmap(y_true, y_pred, y_probs, roc_auc, ap)
        plot_probability_histogram(y_true, y_probs)
        plot_per_class_accuracy(y_true, y_pred)
        plot_misclassified_grid(inf_df, y_true, y_pred, y_probs)
        plot_summary_dashboard(y_true, y_pred, y_probs, roc_auc, ap)
        save_classification_report(y_true, y_pred)
        metrics = save_metrics_json(y_true, y_pred, y_probs, roc_auc, ap)

        if not args.no_gradcam:
            plot_gradcam_samples(model, inf_df, y_pred, y_probs, device)

        print(f"\n{'=' * 65}")
        print("  All outputs saved to: outputs/")
        print(f"{'=' * 65}")
        print(f"  Overall Accuracy  : {metrics['overall_accuracy_%']:.2f}%")
        print(f"  ROC AUC           : {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision     : {metrics['avg_precision_ap']:.4f}")
        print(f"  MCC               : {metrics['mcc']:.4f}")
        print(f"  Cohen Kappa       : {metrics['cohen_kappa']:.4f}")
        print(f"  Sensitivity       : {metrics['sensitivity_%']:.2f}%")
        print(f"  Specificity       : {metrics['specificity_%']:.2f}%")
        print(f"{'=' * 65}\n")
    else:
        plot_predictions_only(y_pred, y_probs)
        if not args.no_gradcam:
            plot_gradcam_samples(model, inf_df, y_pred, y_probs, device)

        print(f"\n{'=' * 65}")
        print("  Prediction-only outputs saved to: outputs/")
        print(f"  Benign predicted    : {(y_pred == 0).sum()}")
        print(f"  Malignant predicted : {(y_pred == 1).sum()}")
        print(f"{'=' * 65}")
        print("\n  For full metrics, provide: --metadata HAM10000_metadata.csv\n")


if __name__ == "__main__":
    main()
