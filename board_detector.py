# board_detector.py
# Detects and crops the chessboard region from an arbitrary image using contour detection.
#
# Standalone:
#   python board_detector.py --image screenshot.png
#   python board_detector.py --image screenshot.png --out board.png
#
# Import:
#   from board_detector import crop_board

import cv2
import numpy as np
import argparse
import os


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _warp(img: np.ndarray, pts: np.ndarray, size: int) -> np.ndarray:
    """Perspective-transforms the quadrilateral defined by pts into a square of given size."""
    src = _order_points(pts)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


def _find_board_contour(gray: np.ndarray, img_area: int):
    """
    Tries multiple Canny thresholds and returns (pts, edges) for the best board
    quadrilateral, or (None, None) if nothing is found.
    """
    for lo, hi in [(30, 100), (50, 150), (80, 200)]:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # RETR_LIST finds all contours, including those inside a UI frame
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:20]:
            area = cv2.contourArea(c)
            # Must be between 5% and 90% of the image — skip screen borders and tiny noise
            if not (img_area * 0.05 < area < img_area * 0.90):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            # Board must be roughly square
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            w = np.linalg.norm(rect[1] - rect[0])
            h = np.linalg.norm(rect[3] - rect[0])
            if max(w, h) / (min(w, h) + 1e-6) < 1.6:
                return pts, edges
    return None, None


def crop_board_debug(image: np.ndarray, size: int = 640):
    """
    Same as crop_board but also returns intermediate debug images.

    Returns
    -------
    board       : perspective-corrected crop (RGB uint8), or None if not found
    edges       : Canny edge image (grayscale uint8)
    contour_vis : original image with detected contour drawn on it (RGB uint8)
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]

    pts, edges = _find_board_contour(gray, img_area)

    contour_vis = bgr.copy()
    if pts is not None:
        cv2.polylines(contour_vis, [pts.astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=3)
        for p in pts.astype(np.int32):
            cv2.circle(contour_vis, tuple(p), 8, (0, 0, 255), -1)
    contour_vis = cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB)

    if pts is None:
        return None, edges, contour_vis

    warped = _warp(bgr, pts, size)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), edges, contour_vis


def crop_board(image: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Finds the chessboard region in the image and returns a perspective-corrected
    top-down crop as an RGB uint8 array.

    Parameters
    ----------
    image : RGB uint8 array
    size  : output square size in pixels (default 640)

    Raises
    ------
    ValueError if no board is found.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]

    pts, _ = _find_board_contour(gray, img_area)
    if pts is None:
        raise ValueError("No chessboard quadrilateral found in image.")

    warped = _warp(bgr, pts, size)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the chessboard from an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--out", default=None, help="Output path (default: <input>_board.png)")
    parser.add_argument("--size", type=int, default=640, help="Output square size in pixels (default: 640)")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    base = os.path.splitext(args.image)[0]

    pts, edges = _find_board_contour(gray, img_area)

    # Debug image 1: Canny edges
    edges_out = base + "_edges.png"
    cv2.imwrite(edges_out, edges)
    print(f"[board_detector] Edges saved      → {edges_out}")

    # Debug image 2: detected contour drawn on original
    contour_vis = img_bgr.copy()
    if pts is not None:
        cv2.polylines(contour_vis, [pts.astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=3)
        for p in pts.astype(np.int32):
            cv2.circle(contour_vis, tuple(p), 8, (0, 0, 255), -1)
    contour_out = base + "_contour.png"
    cv2.imwrite(contour_out, contour_vis)
    print(f"[board_detector] Contour saved    → {contour_out}")

    if pts is None:
        print("[board_detector] No board found — check edges and contour images for clues.")
    else:
        # Final cropped board
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        board = crop_board(img_rgb, size=args.size)
        out = args.out or base + "_board.png"
        cv2.imwrite(out, cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
        print(f"[board_detector] Board saved      → {out}")

