# evaluate.py
# Evalueert de classifier op de testset en slaat confusion matrix op.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import CLASS_NAMES, NUM_CLASSES, PLOTS_DIR, RUN_ID


def plot_confusion_matrix(model, X_test, y_test):
    """Slaat een confusion matrix heatmap op voor de testset."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — test set (run {RUN_ID})")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"confusion_matrix_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix opgeslagen → {save_path}")
