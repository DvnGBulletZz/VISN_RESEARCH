# evaluate.py
# Runs evaluation on the test set and saves a confusion matrix, MAE and mAP plot.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import CLASS_NAMES, GRID_S, NUM_CLASSES, PLOTS_DIR, RUN_ID
from preprocessing import label_encoder
from train import encode_targets


def _extract_predictions(y_true, y_pred):
    """Extracts true and predicted class indices from cells where an object is present."""
    true_labels, pred_labels = [], []
    for i in range(len(y_true)):
        for row in range(GRID_S):
            for col in range(GRID_S):
                if y_true[i, row, col, 4] > 0.5:
                    true_labels.append(np.argmax(y_true[i, row, col, 5:]))
                    pred_labels.append(np.argmax(y_pred[i, row, col, 5:]))
    return true_labels, pred_labels


def plot_confusion_matrix(model, X_test, test_ann):
    """Saves a confusion matrix heatmap for the test set."""
    y_true = encode_targets(test_ann)
    y_pred = model.predict(X_test, verbose=0)
    true_labels, pred_labels = _extract_predictions(y_true, y_pred)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — test set (run {RUN_ID})")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"confusion_matrix_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[evaluate] Confusion matrix saved → {save_path}")


def plot_mae(model, X_test, test_ann):
    """Saves a bar chart of MAE per box coordinate."""
    y_true = encode_targets(test_ann)
    y_pred = model.predict(X_test, verbose=0)
    obj_mask = y_true[..., 4] > 0.5

    mae_per_coord = []
    for c, name in enumerate(['x', 'y', 'w', 'h']):
        mae_per_coord.append({'coord': name,
                              'mae': float(np.mean(np.abs(y_true[..., c][obj_mask] - y_pred[..., c][obj_mask])))})

    df = pd.DataFrame(mae_per_coord)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='coord', y='mae', palette='muted', ax=ax)
    ax.set_title(f"MAE per box coordinate — test set (run {RUN_ID})")
    ax.set_xlabel("Coordinate"); ax.set_ylabel("MAE")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"mae_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[evaluate] MAE plot saved → {save_path}")


def _compute_iou(b1, b2):
    """IoU between two [cx, cy, w, h] normalised boxes."""
    def corners(cx, cy, w, h):
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2
    x1, y1, x2, y2 = corners(*b1)
    x3, y3, x4, y4 = corners(*b2)
    ix = max(0, min(x2, x4) - max(x1, x3))
    iy = max(0, min(y2, y4) - max(y1, y3))
    inter = ix * iy
    union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
    return inter / union if union > 0 else 0.0


def _ap(precisions, recalls):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p) if p else 0.0
    return ap / 11


def plot_map(model, X_test, test_ann, iou_threshold=0.5):
    """Saves per-class AP bar chart."""
    y_true = encode_targets(test_ann)
    y_pred = model.predict(X_test, verbose=0)

    class_results = {c: [] for c in range(NUM_CLASSES)}
    class_n_gt    = {c: 0  for c in range(NUM_CLASSES)}

    for i in range(len(y_true)):
        for row in range(GRID_S):
            for col in range(GRID_S):
                if y_true[i, row, col, 4] < 0.5:
                    continue
                true_cls = int(np.argmax(y_true[i, row, col, 5:]))
                class_n_gt[true_cls] += 1
                conf     = float(y_pred[i, row, col, 4])
                pred_cls = int(np.argmax(y_pred[i, row, col, 5:]))
                iou      = _compute_iou(y_true[i, row, col, :4], y_pred[i, row, col, :4])
                tp = 1 if (iou >= iou_threshold and pred_cls == true_cls) else 0
                class_results[true_cls].append((conf, tp))

    ap_per_class = {}
    for c in range(NUM_CLASSES):
        results = sorted(class_results[c], key=lambda x: -x[0])
        n_gt = class_n_gt[c]
        if n_gt == 0:
            ap_per_class[CLASS_NAMES[c]] = 0.0; continue
        tp_cum = 0
        precs, recs = [], []
        for rank, (_, tp) in enumerate(results, 1):
            tp_cum += tp
            precs.append(tp_cum / rank)
            recs.append(tp_cum / n_gt)
        ap_per_class[CLASS_NAMES[c]] = _ap(precs, recs)

    mean_ap = np.mean(list(ap_per_class.values()))
    df = pd.DataFrame({'class': list(ap_per_class.keys()), 'AP': list(ap_per_class.values())})

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x='class', y='AP', palette='muted', ax=ax)
    ax.axhline(mean_ap, color='red', linestyle='--', linewidth=1, label=f"mAP = {mean_ap:.3f}")
    ax.set_title(f"AP per class @ IoU 0.5 — test set (run {RUN_ID})")
    ax.set_xlabel("Class"); ax.set_ylabel("Average Precision")
    ax.tick_params(axis='x', rotation=45); ax.legend()
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"map_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[evaluate] mAP plot saved → {save_path} | mAP = {mean_ap:.3f}")