"""
evaluate.py  -  Skin Cancer Classification  |  Model Evaluation Script
=======================================================================
Works with a FLAT image folder + HAM10000_metadata.csv for labels.
No benign/malignant subfolders needed.

Generates and saves to  outputs/  folder:
  1. confusion_matrix.png
  2. roc_curve.png
  3. probability_histogram.png
  4. per_class_accuracy.png
  5. summary_dashboard.png
  6. classification_report.txt
  7. metrics_summary.json
  8. predictions_table.csv   (image-by-image predictions)

Usage (zero-argument, uses defaults):
  python evaluate.py

Or with custom paths:
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    precision_recall_fscore_support,
)
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES        = ["Benign", "Malignant"]
MALIGNANT_CLASSES  = {"mel", "bcc", "akiec"}   # same mapping used in training
OUTPUT_DIR         = "outputs"
BATCH_SIZE         = 32
IMAGE_SIZE         = 224

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
# BUILD DATAFRAME from flat folder + metadata CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(img_dir: str, metadata_path: str | None) -> pd.DataFrame:
    """
    Matches images in img_dir to their true labels from HAM10000_metadata.csv.
    Images whose image_id is not found in the CSV are skipped with a warning.
    If no metadata path is provided, predictions are still run but no
    ground-truth metrics (confusion matrix etc.) are produced.
    """
    # Collect all images in the folder
    exts = {".jpg", ".jpeg", ".png"}
    img_files = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if not img_files:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    print(f"[OK] Found {len(img_files)} images in: {img_dir}")

    # Build base dataframe from filenames (strip extension = image_id)
    records = []
    for fname in img_files:
        image_id = os.path.splitext(fname)[0]
        records.append({"image_id": image_id, "filename": fname,
                         "filepath": os.path.join(img_dir, fname)})

    df = pd.DataFrame(records)

    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        print(f"[OK] Metadata loaded : {len(meta)} rows from {metadata_path}")

        # Derive binary label from dx column
        meta["label"] = meta["dx"].apply(
            lambda x: 1 if str(x).lower() in MALIGNANT_CLASSES else 0
        )
        meta["label_name"] = meta["label"].map({0: "Benign", 1: "Malignant"})

        df = df.merge(meta[["image_id", "dx", "label", "label_name"]],
                      on="image_id", how="left")

        missing = df["label"].isna().sum()
        if missing:
            print(f"[!] {missing} image(s) not found in metadata — skipping them.")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        print(f"[OK] Matched {len(df)} images with labels.")
        print(f"     Benign: {(df['label']==0).sum()}  |  Malignant: {(df['label']==1).sum()}")
    else:
        print("[!] No metadata CSV found — running in PREDICTION-ONLY mode.")
        print("    Confusion matrix and classification report will be SKIPPED.")
        df["label"]      = -1
        df["label_name"] = "Unknown"
        df["dx"]         = "Unknown"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class FlatImageDataset(Dataset):
    """Loads images from a flat folder using a dataframe."""

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"]) if row["label"] != -1 else -1
        return image, label, row["image_id"]


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    labels  = [b[1] for b in batch]
    ids     = [b[2] for b in batch]
    return images, labels, ids


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

    all_ids, all_labels, all_preds, all_probs = [], [], [], []

    print("\nRunning inference ...")
    with torch.no_grad():
        for images, labels, ids in tqdm(loader, unit="batch"):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_ids.extend(ids)
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_ids),
        np.array(all_labels, dtype=int),
        np.array(all_preds,  dtype=int),
        np.array(all_probs),             # shape [N, 2]
    )


# ─────────────────────────────────────────────────────────────────────────────
# SAVE PREDICTIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(image_ids, y_true, y_pred, y_probs):
    rows = []
    for img_id, true, pred, prob in zip(image_ids, y_true, y_pred, y_probs):
        rows.append({
            "image_id":          img_id,
            "true_label":        CLASS_NAMES[true]  if true  != -1 else "Unknown",
            "predicted_label":   CLASS_NAMES[pred],
            "prob_benign_%":     round(float(prob[0]) * 100, 2),
            "prob_malignant_%":  round(float(prob[1]) * 100, 2),
            "correct":           (true == pred) if true != -1 else "N/A",
        })
    results_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "predictions_table.csv")
    results_df.to_csv(path, index=False)
    print(f"[OK] Saved -> {path}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold", y=1.02)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"}, ax=axes[0])
    axes[0].set_title("Raw Counts", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
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
    ax.plot(fpr, tpr, color=PAL["accent"], lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PAL["accent"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return save_fig(fig, "roc_curve.png"), roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – PROBABILITY HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_probability_histogram(y_true, y_probs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Predicted Probability Distributions", fontsize=14,
                 fontweight="bold", y=1.02)

    for idx, (cls_name, color) in enumerate(
        zip(CLASS_NAMES, [PAL["Benign"], PAL["Malignant"]])
    ):
        probs_cls = y_probs[y_true == idx, 1]
        axes[idx].hist(probs_cls, bins=40, color=color, edgecolor="white", alpha=0.85)
        axes[idx].axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
        axes[idx].set_title(f"True Class: {cls_name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("P(Malignant)", fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "probability_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – PER-CLASS ACCURACY
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
    x, width, colors = np.arange(len(CLASS_NAMES)), 0.25, ["#6366f1", "#f59e0b", "#10b981"]
    for i, (values, metric) in enumerate(
        zip([prec * 100, rec * 100, f1 * 100], ["Precision", "Recall", "F1-Score"])
    ):
        rects = axes[1].bar(x + i * width, values, width,
                            label=metric, color=colors[i], edgecolor="white")
        for rect, val in zip(rects, values):
            axes[1].text(rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + 0.5,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x + width); axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylim(0, 115); axes[1].set_ylabel("Score (%)", fontsize=11)
    axes[1].set_title("Precision / Recall / F1", fontsize=12)
    axes[1].legend(fontsize=10); axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return save_fig(fig, "per_class_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(y_true, y_pred, y_probs, roc_auc):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Model Evaluation Dashboard  -  Skin Cancer Classification (ResNet50 / HAM10000)",
        fontsize=15, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.42)

    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax1)
    ax1.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=9); ax1.set_ylabel("Actual", fontsize=9)

    # 2. Normalised CM
    ax2 = fig.add_subplot(gs[0, 1])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_n, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax2)
    ax2.set_title("Confusion Matrix (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted", fontsize=9); ax2.set_ylabel("Actual", fontsize=9)

    # 3. ROC
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    ax3.plot(fpr, tpr, color=PAL["accent"], lw=2, label=f"AUC = {roc_auc:.4f}")
    ax3.fill_between(fpr, tpr, alpha=0.08, color=PAL["accent"])
    ax3.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.02)
    ax3.set_xlabel("FPR", fontsize=9); ax3.set_ylabel("TPR", fontsize=9)
    ax3.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(linestyle="--", alpha=0.4)

    # 4. Prob histogram - Benign
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(y_probs[y_true == 0, 1], bins=35, color=PAL["Benign"],
             edgecolor="white", alpha=0.85)
    ax4.axvline(0.5, color="black", linestyle="--", lw=1.5)
    ax4.set_title("P(Malignant) | True Benign", fontsize=11, fontweight="bold")
    ax4.set_xlabel("P(Malignant)", fontsize=9); ax4.set_ylabel("Count", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    # 5. Prob histogram - Malignant
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_probs[y_true == 1, 1], bins=35, color=PAL["Malignant"],
             edgecolor="white", alpha=0.85)
    ax5.axvline(0.5, color="black", linestyle="--", lw=1.5)
    ax5.set_title("P(Malignant) | True Malignant", fontsize=11, fontweight="bold")
    ax5.set_xlabel("P(Malignant)", fontsize=9); ax5.set_ylabel("Count", fontsize=9)
    ax5.grid(axis="y", linestyle="--", alpha=0.4)

    # 6. Precision / Recall / F1
    ax6 = fig.add_subplot(gs[1, 2])
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
# PREDICTION-ONLY PLOT (when no labels available)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_only(y_pred, y_probs):
    """Simple summary chart for unlabelled prediction runs."""
    benign_count    = int((y_pred == 0).sum())
    malignant_count = int((y_pred == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction Summary (no ground-truth labels)",
                 fontsize=13, fontweight="bold")

    # Pie chart
    axes[0].pie([benign_count, malignant_count],
                labels=CLASS_NAMES,
                colors=[PAL["Benign"], PAL["Malignant"]],
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Predicted Class Distribution", fontsize=12)

    # Probability histogram
    axes[1].hist(y_probs[:, 1], bins=40, color=PAL["accent"], edgecolor="white", alpha=0.85)
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
        f.write("Skin Cancer Classification  -  ResNet50 / HAM10000\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
    print(f"[OK] Saved -> {path}")
    print("\n" + "=" * 55)
    print("Classification Report:")
    print("=" * 55)
    print(report)
    return path


def save_metrics_json(y_true, y_pred, y_probs, roc_auc):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    overall_acc = accuracy_score(y_true, y_pred) * 100
    metrics = {
        "overall_accuracy_%": round(overall_acc, 4),
        "roc_auc":            round(float(roc_auc), 4),
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved -> {path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Skin Cancer model on flat image folder + HAM10000 metadata CSV."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="assets/test_imgs",
        help="Flat folder of test images (default: assets/test_imgs).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to HAM10000_metadata.csv. If omitted, runs in prediction-only mode.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="assets/models/best_resnet50_full.pth",
        help="Path to trained .pth model weights.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print("  Skin Cancer Classification  -  Evaluation Script")
    print(f"{'=' * 60}")
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model_path}")
    print(f"  Image dir   : {args.img_dir}")
    print(f"  Metadata    : {args.metadata or '(none — prediction-only mode)'}")
    print(f"  Output dir  : {OUTPUT_DIR}\n")

    make_output_dir()

    # Build dataframe
    df = build_dataframe(args.img_dir, args.metadata)

    # Load model
    model = load_model(args.model_path, device)

    # Run inference
    image_ids, y_true, y_pred, y_probs = run_inference(model, df, device)

    # Save per-image predictions CSV always
    print(f"\n{'=' * 50}")
    print("  Saving outputs ...")
    print(f"{'=' * 50}")
    save_predictions_csv(image_ids, y_true, y_pred, y_probs)

    has_labels = (y_true != -1).all()

    if has_labels:
        # Full evaluation outputs
        plot_confusion_matrix(y_true, y_pred)
        _, roc_auc = plot_roc_curve(y_true, y_probs)
        plot_probability_histogram(y_true, y_probs)
        plot_per_class_accuracy(y_true, y_pred)
        plot_summary_dashboard(y_true, y_pred, y_probs, roc_auc)
        save_classification_report(y_true, y_pred)
        metrics = save_metrics_json(y_true, y_pred, y_probs, roc_auc)

        print(f"\n{'=' * 60}")
        print("  All outputs saved to: outputs/")
        print(f"{'=' * 60}")
        print(f"  Overall Accuracy : {metrics['overall_accuracy_%']:.2f}%")
        print(f"  ROC AUC          : {metrics['roc_auc']:.4f}")
        print(f"{'=' * 60}\n")
    else:
        # Prediction-only outputs
        plot_predictions_only(y_pred, y_probs)
        print(f"\n{'=' * 60}")
        print("  Prediction-only outputs saved to: outputs/")
        print(f"  Benign predicted    : {(y_pred == 0).sum()}")
        print(f"  Malignant predicted : {(y_pred == 1).sum()}")
        print(f"{'=' * 60}\n")
        print("  To get full metrics (confusion matrix, ROC etc.), provide")
        print("  --metadata path/to/HAM10000_metadata.csv\n")


if __name__ == "__main__":
    main()
