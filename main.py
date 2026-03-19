# main.py
# Entry point — keeps as thin as possible.

from data_loader import load_all_annotations, plot_class_distribution, plot_bbox_verification, crop_patches
from preprocessing import preprocess


def main():
    # Load annotations for the training split
    df = load_all_annotations(split="train")

    # Plot class distribution
    plot_class_distribution(df, split="train")

    # Crop patches from source images
    images, labels = crop_patches(df)

    # Verify crops look correct after resize
    plot_bbox_verification(images, labels, n_patches=16)

    # Preprocess: normalise pixels and encode labels
    X, y = preprocess(images, labels)
    print(f"[main] X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()