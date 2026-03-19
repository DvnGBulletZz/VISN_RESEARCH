# data_loader.py
# Responsible for:
#   - Reading the annotation CSVs from both dataset sets
#   - Filtering out the 'board' class (set2)
#   - Normalising class labels via CLASS_MAP
#   - Cropping individual piece patches from source images
#   - Returning a flat list of (image_array, label) tuples ready for preprocessing

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window is opened
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    SPLITS,
    CLASS_MAP, CLASS_NAMES,
    IMG_HEIGHT, IMG_WIDTH,
    PLOTS_DIR, RUN_ID,
)


def load_annotations(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Load a single annotation CSV and attach the full image path.

    Args:
        csv_path  : Path to _annotations.csv
        image_dir : Directory that contains the image files

    Returns:
        DataFrame with columns:
            filename, width, height, class, xmin, ymin, xmax, ymax, image_path
        Rows with class == 'board' are dropped.
        Classes are normalised to full names via CLASS_MAP.
        Rows whose class is not in CLASS_MAP are also dropped.
    """
    df = pd.read_csv(csv_path)

    # Drop the 'board' background annotation that only exists in set2
    df = df[df['class'] != 'board'].copy()

    # Normalise short abbreviations to full class names
    df['class'] = df['class'].map(CLASS_MAP)

    # Drop any row where the class was not recognised
    n_unknown = df['class'].isna().sum()
    if n_unknown > 0:
        print(f"  [WARNING] Dropped {n_unknown} rows with unknown class in {csv_path}")
    df = df.dropna(subset=['class'])

    # Build the full path to each image file
    df['image_path'] = df['filename'].apply(lambda f: os.path.join(image_dir, f))

    return df


def load_all_annotations(split: str = "train") -> pd.DataFrame:
    """
    Load and combine annotations for a given split from both dataset sets.

    Args:
        split : One of 'train', 'valid', 'test' — must match a key in SPLITS

    Returns:
        Single combined DataFrame for the requested split.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLITS.keys())}")

    frames = []
    for csv_path, image_dir in SPLITS[split]:
        print(f"[data_loader] Loading {split} — {csv_path}")
        df = load_annotations(csv_path, image_dir)
        print(f"  {len(df)} annotations across {df['filename'].nunique()} images")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Combined {split} total: {len(combined)} annotations\n")
    return combined


def crop_patches(df: pd.DataFrame) -> tuple[list, list]:
    """
    For every annotation row, open the source image and crop the bounding box
    to produce a small patch for that chess piece.

    Patches are resized to (IMG_HEIGHT, IMG_WIDTH) and converted to RGB.
    Images that cannot be read from disk are skipped with a warning.

    Args:
        df : DataFrame returned by load_all_annotations()

    Returns:
        images : list of np.ndarray, shape (IMG_HEIGHT, IMG_WIDTH, 3), dtype uint8
        labels : list of str, matching class label for each image patch
    """
    images = []
    labels = []
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping patches"):
        img_path = row['image_path']

        if not os.path.exists(img_path):
            missing += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            missing += 1
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Clamp bounding box to image dimensions to avoid out-of-bounds crops
        h, w = img.shape[:2]
        xmin = max(0, int(row['xmin']))
        ymin = max(0, int(row['ymin']))
        xmax = min(w,  int(row['xmax']))
        ymax = min(h,  int(row['ymax']))

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        patch = img[ymin:ymax, xmin:xmax]
        patch = cv2.resize(patch, (IMG_WIDTH, IMG_HEIGHT))

        images.append(patch)
        labels.append(row['class'])

    if missing > 0:
        print(f"  [WARNING] {missing} image files could not be read and were skipped.")

    print(f"[data_loader] Loaded {len(images)} patches total.")
    return images, labels


def plot_class_distribution(df: pd.DataFrame, split: str = "train"):
    """
    Save a bar chart of annotation counts per class to the plots directory.
    The plot is never shown on screen — it is only written to disk.

    Args:
        df    : Combined annotations DataFrame from load_all_annotations()
        split : Split name used in the chart title and filename
    """
    class_counts = df['class'].value_counts().reindex(CLASS_NAMES, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="muted", ax=ax)

    ax.set_title(f"Class distribution — {split} split (run {RUN_ID})", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of annotations")
    ax.tick_params(axis='x', rotation=45)

    # Count labels on top of each bar
    for bar, count in zip(ax.patches, class_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center', va='bottom', fontsize=9,
        )

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"class_distribution_{split}_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[data_loader] Class distribution chart saved → {save_path}")