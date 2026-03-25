# preprocessing.py
# Prepares images and annotations before they go into the model.

import numpy as np
from config import CLASS_NAMES, NUM_CLASSES, IMG_WIDTH
from sklearn.preprocessing import LabelEncoder


# Module-level encoder so train and predict always use the same label order
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)


def normalise_images(images: list) -> np.ndarray:
    """Converts a list of uint8 images to a float32 array scaled to [0, 1]."""
    return np.array(images, dtype=np.float32) / 255.0


def decode_labels(indices) -> np.ndarray:
    """Converts integer indices back to string class labels."""
    return label_encoder.inverse_transform(indices)


def preprocess(images: list, annotations: list) -> tuple[np.ndarray, list]:
    """Normalises images and returns annotations unchanged for now."""
    X = normalise_images(images)
    return X, annotations


def augment(images: np.ndarray, annotations: list) -> tuple[np.ndarray, list]:
    """
    Doubles the training set with horizontal flips + brightness jitter.
    Only call on training data, not on val/test.
    """
    aug_images, aug_annotations = list(images), list(annotations)

    for img, boxes in zip(images, annotations):
        # Horizontal flip
        flipped = img[:, ::-1, :]
        flipped_boxes = [
            {**b, 'xmin': IMG_WIDTH - b['xmax'], 'xmax': IMG_WIDTH - b['xmin']}
            for b in boxes
        ]
        aug_images.append(flipped)
        aug_annotations.append(flipped_boxes)

        # Brightness jitter on the flipped copy
        factor = np.random.uniform(0.75, 1.25)
        bright = np.clip(flipped * factor, 0.0, 1.0)
        aug_images.append(bright)
        aug_annotations.append(flipped_boxes)

    return np.array(aug_images, dtype=np.float32), aug_annotations