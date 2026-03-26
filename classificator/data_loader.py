# data_loader.py
# Laadt annotaties en knipt per bounding box een patch uit het originele beeld.

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import SPLITS, CLASS_MAP, CLASS_NAMES, PATCH_SIZE, PLOTS_DIR, RUN_ID


def _load_annotations(csv_path: str, image_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['class'] != 'board'].copy()
    df['class'] = df['class'].map(CLASS_MAP)
    df = df.dropna(subset=['class'])
    df['image_path'] = df['filename'].apply(lambda f: os.path.join(image_dir, f))
    return df


def load_patches(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Laadt alle annotaties voor de gegeven split, knipt per bounding box
    een patch uit het originele beeld en schaalt naar PATCH_SIZE×PATCH_SIZE.

    Returns:
        patches : np.ndarray (N, PATCH_SIZE, PATCH_SIZE, 3), float32 in [0, 1]
        labels  : np.ndarray (N,), int32 — index in CLASS_NAMES
    """
    label_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    patches = []
    labels  = []
    missing = 0

    for csv_path, image_dir in SPLITS[split]:
        print(f"[data_loader] Laden {split} — {csv_path}")
        df = _load_annotations(csv_path, image_dir)

        for img_path, group in tqdm(df.groupby("image_path"), desc=f"  Crops {split}"):
            if not os.path.exists(img_path):
                missing += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                missing += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            for _, row in group.iterrows():
                x1 = max(0, int(row['xmin']))
                y1 = max(0, int(row['ymin']))
                x2 = min(w, int(row['xmax']))
                y2 = min(h, int(row['ymax']))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))
                patches.append(crop.astype(np.float32) / 255.0)
                labels.append(label_index[row['class']])

    if missing:
        print(f"  [WARNING] {missing} afbeeldingen overgeslagen.")

    print(f"[data_loader] {split}: {len(patches)} patches geladen.\n")
    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32)


def plot_patch_verification(patches: np.ndarray, labels: np.ndarray, n_per_class: int = 4):
    """
    Slaat een grid op met voorbeeldpatches per klasse zodat je kunt
    verifiëren of de crops correct zijn uitgeknipt en gelabeld.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(n_per_class * 2, len(CLASS_NAMES) * 2))

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_label = CLASS_NAMES.index(class_name)
        idx = np.where(labels == class_label)[0]
        samples = idx[:n_per_class]

        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx]
            if col_idx < len(samples):
                ax.imshow(patches[samples[col_idx]])
            else:
                ax.imshow(np.zeros((PATCH_SIZE, PATCH_SIZE, 3)))
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=7, rotation=0,
                              labelpad=60, va='center')

    plt.suptitle(f"Patch verificatie — train split (run {RUN_ID})", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"patch_verification_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[data_loader] Patch verificatie opgeslagen → {save_path}")


def plot_class_distribution(labels: np.ndarray, split: str = "train"):
    """Slaat een staafdiagram op met het aantal patches per klasse."""
    import seaborn as sns

    os.makedirs(PLOTS_DIR, exist_ok=True)

    counts = pd.Series(labels).value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    counts.index = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="muted", ax=ax)
    ax.set_title(f"Class distributie — {split} split (run {RUN_ID})", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Aantal patches")
    ax.tick_params(axis='x', rotation=45)

    for bar, count in zip(ax.patches, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"class_distribution_{split}_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[data_loader] Class distributie opgeslagen → {save_path}")
