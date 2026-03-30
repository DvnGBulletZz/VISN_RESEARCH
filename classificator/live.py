# live.py
# Live screen capture -> board detection -> cell classification.
# Shows a real-time window with the detected board and piece labels.
#
# Usage:
#   python live.py
#   python live.py --model outputs/models/classifier_run18.h5
#   python live.py --interval 0.5   (seconds between captures, default 1.0)
#
# Controls:
#   Q  - quit
#   S  - save current frame to outputs/live_snapshot.png

import sys
import os
import time
import argparse

import numpy as np
import cv2
import mss
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from board_detector import crop_board_debug
from predict import has_piece, classify_cells, PIECE_SYMBOLS
from config import CLASS_NAMES, PATCH_SIZE, MODEL_SAVE_PATH, PLOTS_DIR


BOARD_SIZE   = 640   # size of the cropped board display
CELL_SIZE    = BOARD_SIZE // 8
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def _split_cells(board_bgr: np.ndarray) -> list:
    cells = []
    for row in range(8):
        for col in range(8):
            y1, x1 = row * CELL_SIZE, col * CELL_SIZE
            cell = board_bgr[y1:y1+CELL_SIZE, x1:x1+CELL_SIZE]
            cells.append(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
    return cells


def _draw_results(board_bgr: np.ndarray, results: list, occupied: list) -> np.ndarray:
    out = board_bgr.copy()
    for i, result in enumerate(results):
        row, col = divmod(i, 8)
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        cx = x1 + CELL_SIZE // 2
        cy = y1 + CELL_SIZE // 2

        color = (0, 200, 0) if occupied[i] else (80, 80, 80)
        cv2.rectangle(out, (x1, y1), (x1 + CELL_SIZE, y1 + CELL_SIZE), color, 1)

        if result is not None:
            label, conf = result
            symbol = PIECE_SYMBOLS.get(label, label)
            is_black = label.startswith('black')
            bg   = (20,  20,  20)  if is_black else (240, 240, 240)
            text = (240, 240, 240) if is_black else (20,  20,  20)

            # Background rectangle for text
            (tw, th), _ = cv2.getTextSize(symbol, FONT, 0.45, 1)
            cv2.rectangle(out,
                          (cx - tw//2 - 2, cy - th - 2),
                          (cx + tw//2 + 2, cy + 4), bg, -1)
            cv2.putText(out, symbol, (cx - tw//2, cy), FONT, 0.45, text, 1, cv2.LINE_AA)

            # Confidence below
            conf_str = f"{conf:.2f}"
            (cw, _), _ = cv2.getTextSize(conf_str, FONT, 0.32, 1)
            cv2.putText(out, conf_str, (cx - cw//2, cy + 12),
                        FONT, 0.32, color, 1, cv2.LINE_AA)
    return out


def _make_debug_strip(cells: list, occupied: list) -> np.ndarray:
    """Small strip showing all 64 cells with green/red border."""
    thumb = 40
    border = 2
    tile = thumb + border * 2
    strip = np.zeros((8 * tile, 8 * tile, 3), dtype=np.uint8)
    for i, (cell, occ) in enumerate(zip(cells, occupied)):
        r, c = divmod(i, 8)
        img = cv2.resize(cv2.cvtColor(cell, cv2.COLOR_RGB2BGR), (thumb, thumb))
        color = (0, 180, 0) if occ else (0, 0, 180)
        img = cv2.copyMakeBorder(img, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=color)
        strip[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = img
    return strip


def run(model_path: str, interval: float):
    print(f"[live] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[live] Model loaded. Press Q to quit, S to save snapshot.")

    sct = mss.mss()
    monitor = sct.monitors[1]  # full primary screen

    window_name  = "Live board - classificator"
    debug_name   = "Cell detection"
    cv2.namedWindow(window_name,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_name,   cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, BOARD_SIZE, BOARD_SIZE)
    cv2.resizeWindow(debug_name,  320, 320)

    last_time   = 0
    last_board  = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    last_debug  = np.zeros((320, 320, 3), dtype=np.uint8)
    status      = "Waiting..."

    while True:
        now = time.time()

        if now - last_time >= interval:
            last_time = now

            # Capture screen
            raw = np.array(sct.grab(monitor))
            frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

            # Detect board
            board_rgb, edges, _ = crop_board_debug(frame_rgb, size=BOARD_SIZE)

            if board_rgb is None:
                status = "No board detected"
            else:
                board_bgr = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)
                cells = _split_cells(board_bgr)

                # Piece detection
                occupied, all_edges, all_thresh = [], [], []
                for cell in cells:
                    occ, e, t = has_piece(cell)
                    occupied.append(occ)
                    all_edges.append(e)
                    all_thresh.append(t)

                # Classify
                results  = classify_cells(model, cells, occupied)
                n_pieces = sum(1 for r in results if r is not None)
                status   = f"{n_pieces} pieces | {sum(occupied)} cells occupied"

                last_board = _draw_results(board_bgr, results, occupied)
                last_debug = cv2.resize(_make_debug_strip(cells, occupied), (320, 320))

        # Overlay status text
        display = last_board.copy()
        cv2.putText(display, status, (6, 20), FONT, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(display, f"interval: {interval}s  |  Q=quit  S=save",
                    (6, BOARD_SIZE - 8), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        cv2.imshow(debug_name,  last_debug)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(PLOTS_DIR, exist_ok=True)
            snap_path = os.path.join(PLOTS_DIR, "live_snapshot.png")
            cv2.imwrite(snap_path, display)
            print(f"[live] Snapshot saved -> {snap_path}")

    cv2.destroyAllWindows()
    sct.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live chess board classifier.")
    parser.add_argument("--model",    default=MODEL_SAVE_PATH, help="Path to classifier .h5")
    parser.add_argument("--interval", type=float, default=1.0,  help="Seconds between captures")
    args = parser.parse_args()

    run(model_path=args.model, interval=args.interval)
