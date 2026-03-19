# data_loader.py
# Loads full board images with all their annotations.
# No more patch cropping — the full image goes into the model.

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from config import (
    SPLITS, CLASS_MAP, CLASS_NAMES,
    IMG_HEIGHT, IMG_WIDTH,
    PLOTS_DIR, RUN_ID,
)


def load_annotations(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Reads one annotation CSV, drops 'board' rows and normalises class names."""
    df = pd.read_csv(csv_path)

    df = df[df['class'] != 'board'].copy()
    df['class'] = df['class'].map(CLASS_MAP)

    n_unknown = df['class'].isna().sum()
    if n_unknown > 0:
        print(f"  [WARNING] Dropped {n_unknown} rows with unknown class in {csv_path}")
    df = df.dropna(subset=['class'])

    df['image_path'] = df['filename'].apply(lambda f: os.path.join(image_dir, f))
    return df


def load_all_annotations(split: str = "train") -> pd.DataFrame:
    """Loads and combines annotations from both datasets for the given split."""
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


def load_images(df: pd.DataFrame) -> tuple[list, list]:
    """
    Loads full board images and their annotation boxes.
    Returns one entry per unique image, each with a list of all its boxes.

    Returns:
        images     : list of np.ndarray (IMG_HEIGHT, IMG_WIDTH, 3), dtype uint8
        annotations: list of lists, each inner list contains dicts with
                     keys: class, xmin, ymin, xmax, ymax (scaled to IMG size)
    """
    images      = []
    annotations = []
    missing     = 0

    unique_paths = df['image_path'].unique()

    for img_path in tqdm(unique_paths, desc="Loading images"):
        if not os.path.exists(img_path):
            missing += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            missing += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Scale factors to adjust box coordinates after resize
        scale_x = IMG_WIDTH  / orig_w
        scale_y = IMG_HEIGHT / orig_h

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Collect all boxes for this image and scale their coordinates
        rows = df[df['image_path'] == img_path]
        boxes = []
        for _, row in rows.iterrows():
            boxes.append({
                'class': row['class'],
                'xmin':  int(row['xmin'] * scale_x),
                'ymin':  int(row['ymin'] * scale_y),
                'xmax':  int(row['xmax'] * scale_x),
                'ymax':  int(row['ymax'] * scale_y),
            })

        images.append(img)
        annotations.append(boxes)

    if missing > 0:
        print(f"  [WARNING] {missing} images could not be read and were skipped.")

    print(f"[data_loader] Loaded {len(images)} images total.")
    return images, annotations


def plot_bbox_verification(images: list, annotations: list, n_images: int = 4):
    """
    Saves a grid of full board images with all bounding boxes drawn on them
    after resize so you can verify the box scaling is correct.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = min(n_images, len(images))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, boxes in zip(axes, images[:n], annotations[:n]):
        ax.imshow(img)
        for box in boxes:
            xmin, ymin = box['xmin'], box['ymin']
            w = box['xmax'] - xmin
            h = box['ymax'] - ymin
            rect = mpatches.Rectangle(
                (xmin, ymin), w, h,
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 4, box['class'], color='lime', fontsize=6)
        ax.axis('off')

    plt.suptitle(f"Bounding box verification after resize (run {RUN_ID})", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"bbox_verification_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[data_loader] Bounding box verification saved → {save_path}")


def plot_class_distribution(df: pd.DataFrame, split: str = "train"):
    """Saves a bar chart of annotation counts per class."""
    class_counts = df['class'].value_counts().reindex(CLASS_NAMES, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="muted", ax=ax)

    ax.set_title(f"Class distribution — {split} split (run {RUN_ID})", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of annotations")
    ax.tick_params(axis='x', rotation=45)

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