# preprocessing.py
# Prepares images and annotations before they go into the model.

import numpy as np
from config import CLASS_NAMES, NUM_CLASSES
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