# data_loader.py
# Responsible for:
#   - Reading the annotation CSVs from both dataset sets
#   - Filtering out the 'board' class (set2)
#   - Normalising class labels via CLASS_MAP
#   - Cropping individual piece patches from source images
#   - Saving a bounding box visualisation to verify crops are correct
#   - Plotting and saving the class distribution chart

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window is opened
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from config import *



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


def crop_patches(df: pd.DataFrame) -> tuple[list, list]:
    """Crops bounding box patches from source images and returns (images, labels)."""
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        xmin = max(0, int(row['xmin']))
        ymin = max(0, int(row['ymin']))
        xmax = min(w,  int(row['xmax']))
        ymax = min(h,  int(row['ymax']))

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


def plot_bbox_verification(images: list, labels: list, n_patches: int = 16):
    """
    Saves a grid of resized 64x64 patches with a bounding box drawn around
    each one so you can verify the crop and resize came out correctly.

    Args:
        images   : list of resized patches from crop_patches()
        labels   : matching class labels
        n_patches: how many patches to show in the grid
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = min(n_patches, len(images))
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i])

            # Draw a box around the full patch — the patch IS the crop so the
            # box confirms the resize kept the piece roughly centred
            rect = mpatches.Rectangle(
                (0, 0), IMG_WIDTH - 1, IMG_HEIGHT - 1,
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(labels[i], fontsize=6)
        ax.axis('off')

    plt.suptitle(f"Patch verification after crop & resize (run {RUN_ID})", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"bbox_verification_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[data_loader] Patch verification saved → {save_path}")


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