# evaluate.py
# Runs evaluation on the test set and saves a confusion matrix and MAE plot.

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
    """
    Pulls the predicted and true class labels from the grid tensors.
    Only looks at cells where an object is present (confidence > 0.5).
    Returns two flat lists: true labels and predicted labels.
    """
    true_labels = []
    pred_labels = []

    for i in range(len(y_true)):
        for row in range(GRID_S):
            for col in range(GRID_S):
                if y_true[i, row, col, 4] > 0.5:
                    true_idx = np.argmax(y_true[i, row, col, 5:])
                    pred_idx = np.argmax(y_pred[i, row, col, 5:])
                    true_labels.append(true_idx)
                    pred_labels.append(pred_idx)

    return true_labels, pred_labels


def plot_confusion_matrix(model, X_test, test_ann):
    """Saves a confusion matrix heatmap for the test set."""
    y_true_enc = encode_targets(test_ann)
    y_pred     = model.predict(X_test, verbose=0)

    true_labels, pred_labels = _extract_predictions(y_true_enc, y_pred)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap='Blues', ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — test set (run {RUN_ID})")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"confusion_matrix_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {save_path}")


def plot_mae(model, X_test, test_ann):
    """
    Saves a bar chart of mean absolute error per box coordinate
    (x, y, w, h) across all cells that contain an object.
    """
    y_true = encode_targets(test_ann)
    y_pred = model.predict(X_test, verbose=0)

    obj_mask = y_true[..., 4] > 0.5
    mae_per_coord = []

    for c, name in enumerate(['x', 'y', 'w', 'h']):
        true_vals = y_true[..., c][obj_mask]
        pred_vals = y_pred[..., c][obj_mask]
        mae_per_coord.append({'coord': name, 'mae': float(np.mean(np.abs(true_vals - pred_vals)))})

    df = pd.DataFrame(mae_per_coord)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='coord', y='mae', palette='muted', ax=ax)
    ax.set_title(f"MAE per box coordinate — test set (run {RUN_ID})")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("MAE")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"mae_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] MAE plot saved → {save_path}")