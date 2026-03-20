# predict.py
# Runs inference on two example images and saves them with predicted bounding boxes.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import CLASS_NAMES, GRID_S, IMG_HEIGHT, IMG_WIDTH, PLOTS_DIR, RUN_ID
from preprocessing import label_encoder


def predict_boxes(model, image: np.ndarray, conf_threshold: float = 0.5) -> list:
    """Runs the model on a single normalised image and returns predicted boxes above threshold."""
    inp  = np.expand_dims(image, axis=0)
    pred = model.predict(inp, verbose=0)[0]

    boxes = []
    for row in range(GRID_S):
        for col in range(GRID_S):
            conf = pred[row, col, 4]
            if conf < conf_threshold:
                continue
            x_cell, y_cell, w_norm, h_norm = pred[row, col, :4]
            cx = (col + x_cell) / GRID_S * IMG_WIDTH
            cy = (row + y_cell) / GRID_S * IMG_HEIGHT
            w  = w_norm * IMG_WIDTH
            h  = h_norm * IMG_HEIGHT
            class_idx  = np.argmax(pred[row, col, 5:])
            class_name = label_encoder.inverse_transform([class_idx])[0]
            boxes.append({
                'class': class_name, 'confidence': float(conf),
                'xmin': int(cx - w/2), 'ymin': int(cy - h/2),
                'xmax': int(cx + w/2), 'ymax': int(cy + h/2),
            })
    return boxes


def plot_predictions(model, images: list, n: int = 2):
    """Saves a figure with n example images and predicted bounding boxes."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, img in zip(axes, images[:n]):
        display = (img * 255).astype(np.uint8)
        ax.imshow(display)
        for box in predict_boxes(model, img):
            w = box['xmax'] - box['xmin']
            h = box['ymax'] - box['ymin']
            rect = mpatches.Rectangle(
                (box['xmin'], box['ymin']), w, h,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box['xmin'], box['ymin'] - 4,
                    f"{box['class']} {box['confidence']:.2f}",
                    color='red', fontsize=6)
        ax.axis('off')

    plt.suptitle(f"Predicted bounding boxes (run {RUN_ID})", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"predictions_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[predict] Predictions saved → {save_path}")