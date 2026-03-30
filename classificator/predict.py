# predict.py
# Uses board_detector to crop the board, splits it into 64 cells,
# checks each cell for a piece via contour/edge detection,
# and classifies occupied cells with the trained classifier model.
#
# Usage:
#   python predict.py --image path/to/image.png
#   python predict.py --image path/to/image.png --model outputs/models/classifier_run15.h5
#   python predict.py --image path/to/image.png --output result.png --debug

import sys
import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)   # for board_detector
sys.path.insert(0, _HERE)   # local config must shadow root config

from board_detector import crop_board_debug
from config import CLASS_NAMES, PATCH_SIZE, MODEL_SAVE_PATH, PLOTS_DIR, RUN_ID


PIECE_SYMBOLS = {
    'black-bishop': 'bb', 'black-king':   'bk', 'black-knight': 'bn',
    'black-pawn':   'bp', 'black-queen':  'bq', 'black-rook':   'br',
    'white-bishop': 'wb', 'white-king':   'wk', 'white-knight': 'wn',
    'white-pawn':   'wp', 'white-queen':  'wq', 'white-rook':   'wr',
}

# --- Piece detection thresholds (tune if needed) --------------------------
# Edge density: fraction of pixels that are edges (Canny).
# Empty squares are flat; pieces add outlines and internal detail.
EDGE_DENSITY_THRESHOLD = 0.06

# Minimum contour area (as fraction of cell area) for a blob to count.
# Filters out tiny noise specks.
CONTOUR_AREA_MIN_FRAC = 0.04

# The center crop fraction used for contour check (ignore border artifacts).
CENTER_CROP = 0.75


def split_board_into_cells(board_rgb: np.ndarray) -> list[np.ndarray]:
    """Splits the board into 64 raw cells (row-major). Returns uint8 RGB."""
    size = board_rgb.shape[0]
    cell_size = size // 8
    cells = []
    for row in range(8):
        for col in range(8):
            y1 = row * cell_size
            y2 = y1 + cell_size
            x1 = col * cell_size
            x2 = x1 + cell_size
            cells.append(board_rgb[y1:y2, x1:x2].copy())
    return cells


def has_piece(cell_rgb: np.ndarray):
    """Determines if a cell contains a piece using edge density and contour checks."""
    gray = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    cell_area = h * w

    # --- Edge density check ---
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edge_density = np.count_nonzero(edges) / cell_area

    # --- Adaptive threshold blob check in center region ---
    margin = int(h * (1 - CENTER_CROP) / 2)
    center = gray[margin:h - margin, margin:w - margin]
    thresh_center = cv2.adaptiveThreshold(
        center, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )
    # Pad thresh back to full cell size for the debug grid
    thresh_full = np.zeros_like(gray)
    thresh_full[margin:h - margin, margin:w - margin] = thresh_center

    contours, _ = cv2.findContours(thresh_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_area = thresh_center.shape[0] * thresh_center.shape[1]
    blob_found = any(cv2.contourArea(c) / center_area >= CONTOUR_AREA_MIN_FRAC
                     for c in contours)

    occupied = edge_density >= EDGE_DENSITY_THRESHOLD or blob_found
    return occupied, edges, thresh_full


def classify_cells(model, cells: list, occupied: list) -> list:
    """
    Classifies only the occupied cells, returns (label, confidence) or None.
    Cells flagged as empty are returned as None without touching the model.
    """
    indices = [i for i, occ in enumerate(occupied) if occ]
    results = [None] * len(cells)

    if not indices:
        return results

    # Resize to PATCH_SIZE and normalise in one batch
    patches = []
    for i in indices:
        patch = cv2.resize(cells[i], (PATCH_SIZE, PATCH_SIZE))
        patches.append(patch.astype(np.float32) / 255.0)

    batch = np.array(patches, dtype=np.float32)
    preds = model.predict(batch, verbose=0)

    for batch_idx, cell_idx in enumerate(indices):
        conf = float(np.max(preds[batch_idx]))
        label = CLASS_NAMES[int(np.argmax(preds[batch_idx]))]
        results[cell_idx] = (label, conf)

    return results


def draw_board(board_rgb: np.ndarray, results: list, occupied: list[bool],
               output_path: str, debug: bool = False):
    """Overlays classification results and optional piece-detection markers."""
    size = board_rgb.shape[0]
    cell_size = size // 8

    if debug:
        # Side-by-side: board + occupancy map
        _, axes = plt.subplots(1, 2, figsize=(16, 8))
        ax, ax2 = axes
        occ_img = np.zeros((8, 8, 3), dtype=np.uint8)
        for i, occ in enumerate(occupied):
            r, c = divmod(i, 8)
            occ_img[r, c] = (0, 200, 0) if occ else (40, 40, 40)
        ax2.imshow(occ_img, interpolation='nearest')
        ax2.set_title("Piece detection map (green = piece)", fontsize=10)
        ax2.set_xticks(range(8)); ax2.set_yticks(range(8))
        ax2.set_xticklabels(list("abcdefgh"))
        ax2.set_yticklabels(range(8, 0, -1))
    else:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(board_rgb)

    for i, result in enumerate(results):
        row, col = divmod(i, 8)
        cx = col * cell_size + cell_size // 2
        cy = row * cell_size + cell_size // 2

        edge_color = 'lime' if occupied[i] else 'gray'
        rect = mpatches.Rectangle(
            (col * cell_size, row * cell_size),
            cell_size, cell_size,
            linewidth=1, edgecolor=edge_color, facecolor='none'
        )
        ax.add_patch(rect)

        if result is not None:
            label, conf = result
            symbol = PIECE_SYMBOLS.get(label, label)
            text_color = 'white' if label.startswith('black') else 'black'
            bg_color   = 'black' if label.startswith('black') else 'white'
            ax.text(
                cx, cy, f"{symbol}\n{conf:.2f}",
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color=text_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, alpha=0.75)
            )

    ax.axis('off')
    piece_count = sum(1 for r in results if r is not None)
    ax.set_title(f"Chess board - {piece_count} pieces classified", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[predict] Result saved -> {output_path}")


def _save_internals_grid(cells_gray_imgs: list, occupied: list, name: str):
    """
    Saves an 8x8 grid of grayscale internal images (edges or thresh).
    Green border = piece, red border = empty.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    thumb = PATCH_SIZE
    border = 4
    tile = thumb + border * 2
    grid = np.zeros((8 * tile, 8 * tile), dtype=np.uint8)

    for i, (img_gray, occ) in enumerate(zip(cells_gray_imgs, occupied)):
        row, col = divmod(i, 8)
        img = cv2.resize(img_gray, (thumb, thumb))
        y1 = row * tile + border
        x1 = col * tile + border
        grid[y1:y1 + thumb, x1:x1 + thumb] = img

    # Convert to BGR so we can draw colored borders
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    for i, occ in enumerate(occupied):
        row, col = divmod(i, 8)
        color = (0, 200, 0) if occ else (0, 0, 200)
        y1 = row * tile
        x1 = col * tile
        cv2.rectangle(grid_bgr, (x1, y1), (x1 + tile - 1, y1 + tile - 1), color, border)

    save_path = os.path.join(PLOTS_DIR, f"predict_{name}_run{RUN_ID}.png")
    cv2.imwrite(save_path, grid_bgr)
    print(f"[predict] {name} grid saved   -> {save_path}")


def _save_cell_detection_grid(cells: list, occupied: list):
    """
    Saves an 8x8 grid of all 64 cell images.
    Green border = piece detected, red border = empty.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    thumb = PATCH_SIZE  # each cell thumbnail size
    border = 4
    tile = thumb + border * 2
    grid = np.zeros((8 * tile, 8 * tile, 3), dtype=np.uint8)

    for i, (cell, occ) in enumerate(zip(cells, occupied)):
        row, col = divmod(i, 8)
        img = cv2.resize(cell, (thumb, thumb))
        # border color: green if piece, red if empty
        color = (0, 200, 0) if occ else (200, 0, 0)
        img = cv2.copyMakeBorder(img, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=color)
        y1 = row * tile
        x1 = col * tile
        grid[y1:y1 + tile, x1:x1 + tile] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Add column letters and row numbers
    label_img = np.zeros((8 * tile + 20, 8 * tile + 20, 3), dtype=np.uint8)
    label_img[20:, 20:] = grid
    for c in range(8):
        cv2.putText(label_img, "abcdefgh"[c],
                    (20 + c * tile + tile // 2 - 5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for r in range(8):
        cv2.putText(label_img, str(8 - r),
                    (4, 20 + r * tile + tile // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    save_path = os.path.join(PLOTS_DIR, f"predict_cells_run{RUN_ID}.png")
    cv2.imwrite(save_path, label_img)
    print(f"[predict] Cell detection grid -> {save_path}")


def _save_debug_images(edges, contour_vis, board_rgb):
    """Saves intermediate board-detection images to PLOTS_DIR."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Edges (grayscale)
    edges_path = os.path.join(PLOTS_DIR, f"predict_edges_run{RUN_ID}.png")
    cv2.imwrite(edges_path, edges)
    print(f"[predict] Edges saved        -> {edges_path}")

    # Contour overlay
    contour_path = os.path.join(PLOTS_DIR, f"predict_contour_run{RUN_ID}.png")
    cv2.imwrite(contour_path, cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))
    print(f"[predict] Contour saved      -> {contour_path}")

    # Cropped board
    board_path = os.path.join(PLOTS_DIR, f"predict_board_run{RUN_ID}.png")
    cv2.imwrite(board_path, cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR))
    print(f"[predict] Cropped board saved -> {board_path}")


def predict(image_path: str, model_path: str, output_path: str = None, debug: bool = False):
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Detect and crop board, save debug images
    print("[predict] Detecting board...")
    board_rgb, edges, contour_vis = crop_board_debug(img_rgb, size=512)
    if board_rgb is None:
        raise ValueError("No chessboard found in image.")
    print("[predict] Board detected and cropped to 512x512.")
    _save_debug_images(edges, contour_vis, board_rgb)

    # Step 2: Split into 64 cells
    cells = split_board_into_cells(board_rgb)

    # Step 3: Check each cell for a piece via contours/edges
    print("[predict] Checking cells for pieces...")
    occupied, all_edges, all_thresh = [], [], []
    for cell in cells:
        occ, edges_img, thresh_img = has_piece(cell)
        occupied.append(occ)
        all_edges.append(edges_img)
        all_thresh.append(thresh_img)
    occ_count = sum(occupied)
    print(f"[predict] {occ_count}/64 cells flagged as occupied.")
    _save_cell_detection_grid(cells, occupied)
    _save_internals_grid(all_edges,  occupied, "cells_edges")
    _save_internals_grid(all_thresh, occupied, "cells_thresh")

    # Step 4: Classify occupied cells
    print(f"[predict] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("[predict] Classifying occupied cells...")
    results = classify_cells(model, cells, occupied)
    piece_count = sum(1 for r in results if r is not None)
    print(f"[predict] {piece_count} pieces classified.")

    # Step 5: Draw output
    if output_path is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        output_path = os.path.join(PLOTS_DIR, f"predict_classified_run{RUN_ID}.png")

    draw_board(board_rgb, results, occupied, output_path, debug=debug)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify chess pieces per board cell.")
    parser.add_argument("--image",  required=True,           help="Path to input image")
    parser.add_argument("--model",  default=MODEL_SAVE_PATH, help="Path to classifier .h5 model")
    parser.add_argument("--output", default=None,            help="Output image path")
    parser.add_argument("--debug",  action="store_true",     help="Also save piece-detection map")
    args = parser.parse_args()

    predict(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        debug=args.debug,
    )
