# predict.py
# Runs inference on images and saves them with predicted bounding boxes.
#
# Usage from main.py:
#   plot_predictions(model, X_test, n=2)
#
# Standalone usage (custom image):
#   python predict.py --image path/to/image.jpg
#   python predict.py --image path/to/image.jpg --model outputs/models/model_run8.h5
#   python predict.py --image path/to/image.jpg --threshold 0.4 --output result.png

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from config import GRID_S, IMG_HEIGHT, IMG_WIDTH, PLOTS_DIR, MODELS_DIR, RUN_ID
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
            cx, cy, w_cell, h_cell = pred[row, col, :4]
            cx = (col + cx) / GRID_S * IMG_WIDTH
            cy = (row + cy) / GRID_S * IMG_HEIGHT
            # w/h encoded as w*7 — divide by 7 to get image fraction
            w  = (w_cell / 7) * IMG_WIDTH
            h  = (h_cell / 7) * IMG_HEIGHT
            class_idx  = np.argmax(pred[row, col, 5:])
            class_name = label_encoder.inverse_transform([class_idx])[0]
            boxes.append({
                'class': class_name, 'confidence': float(conf),
                'xmin': int(cx - w/2), 'ymin': int(cy - h/2),
                'xmax': int(cx + w/2), 'ymax': int(cy + h/2),
            })
    return boxes


def _draw_boxes(ax, image: np.ndarray, boxes: list, title: str = None):
    """Draws bounding boxes on an image. Accepts float [0,1] or uint8 [0,255]."""
    display = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    ax.imshow(display)
    for box in boxes:
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
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis('off')


def plot_predictions(model, images: list, n: int = 2):
    """Called from main.py. Saves a figure with n example images and predicted bounding boxes."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = min(n, len(images))
    _, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, img in zip(axes, images[:n]):
        boxes = predict_boxes(model, img)
        _draw_boxes(ax, img, boxes)

    plt.suptitle(f"Predicted bounding boxes (run {RUN_ID})", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"predictions_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[predict] Predictions saved → {save_path}")


def _load_original(image_path: str) -> np.ndarray:
    """Loads an image at its original resolution as an RGB uint8 array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _preprocess_for_model(img_rgb: np.ndarray) -> np.ndarray:
    """Resizes to IMG_HEIGHT×IMG_WIDTH and normalises to [0, 1]."""
    img = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32) / 255.0


def _scale_boxes(boxes: list, orig_w: int, orig_h: int) -> list:
    """Scales boxes from 224×224 model space back to original image dimensions."""
    sx = orig_w / IMG_WIDTH
    sy = orig_h / IMG_HEIGHT
    return [
        {**b,
         'xmin': int(b['xmin'] * sx), 'ymin': int(b['ymin'] * sy),
         'xmax': int(b['xmax'] * sx), 'ymax': int(b['ymax'] * sy)}
        for b in boxes
    ]


def predict_single_image(image_path: str, model_path: str = None,
                          conf_threshold: float = 0.5, output_path: str = None,
                          detect_board: bool = False):
    """
    Loads a model and a custom image, runs detection, and saves the result.
    Bounding boxes are drawn on the original full-resolution image.

    Parameters
    ----------
    image_path    : path to the input image (jpg, png, etc.)
    model_path    : path to the saved model (.h5). Defaults to outputs/models/best_model_run8.h5
    conf_threshold: minimum confidence for a detection (default 0.5)
    output_path   : where to save the result. Defaults to <input>_predicted.png next to the input.
    detect_board  : if True, auto-crop the board region before running the model
    """
    import tensorflow as tf

    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"best_model_run{RUN_ID}.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[predict] Loading model  → {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"[predict] Loading image  → {image_path}")
    original = _load_original(image_path)
    base, ext = os.path.splitext(image_path)
    ext = ext if ext else ".png"

    if detect_board:
        from board_detector import crop_board_debug
        import cv2 as _cv2
        print("[predict] Detecting board region...")
        board, edges, contour_vis = crop_board_debug(original)

        edges_path   = base + "_edges.png"
        contour_path = base + "_contour.png"
        board_path   = base + "_board.png"

        _cv2.imwrite(edges_path, edges)
        _cv2.imwrite(contour_path, _cv2.cvtColor(contour_vis, _cv2.COLOR_RGB2BGR))
        print(f"[predict] Edges saved    → {edges_path}")
        print(f"[predict] Contour saved  → {contour_path}")

        if board is None:
            print("[predict] No board detected — cannot run model.")
            return
        _cv2.imwrite(board_path, _cv2.cvtColor(board, _cv2.COLOR_RGB2BGR))
        print(f"[predict] Board saved    → {board_path}")
        original = board

    orig_h, orig_w = original.shape[:2]

    # Resize and normalise for model input, keep original for display
    model_input = _preprocess_for_model(original)

    # Boxes are returned in 224×224 pixel coordinates — scale back to original
    boxes_model = predict_boxes(model, model_input, conf_threshold=conf_threshold)
    boxes = _scale_boxes(boxes_model, orig_w, orig_h)

    print(f"[predict] {len(boxes)} detection(s) above threshold {conf_threshold}")
    for b in boxes:
        print(f"          {b['class']:15s}  conf={b['confidence']:.2f}  "
              f"box=({b['xmin']},{b['ymin']}) → ({b['xmax']},{b['ymax']})")

    # Figsize proportional to original image
    fig_w = max(7, orig_w / 100)
    fig_h = max(7, orig_h / 100)
    _, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    _draw_boxes(ax, original, boxes, title=os.path.basename(image_path))
    plt.tight_layout()

    if output_path is None:
        output_path = base + "_predicted" + ext

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[predict] Result saved   → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run chess piece detection on a custom image."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image (e.g. photos/my_board.jpg)"
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Path to the model (.h5). Default: outputs/models/best_model_run8.h5"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum confidence for a detection (default: 0.5)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Default: <input>_predicted.png next to the input file"
    )
    parser.add_argument(
        "--detect", action="store_true",
        help="Auto-crop the board region from the image before running the model"
    )
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        model_path=args.model,
        conf_threshold=args.threshold,
        output_path=args.output,
        detect_board=args.detect,
    )
