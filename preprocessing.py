# preprocessing.py
# Handles all data preparation before the data goes into the model:
#   - Normalising pixel values to [0, 1]
#   - Encoding string labels to integers
#   - Converting lists to numpy arrays

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import CLASS_NAMES


# Single encoder instance so train and predict use the same label order
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)


def normalise_images(images: list) -> np.ndarray:
    """Converts a list of uint8 patches to a float32 array scaled to [0, 1]."""
    arr = np.array(images, dtype=np.float32)
    arr /= 255.0
    return arr


def encode_labels(labels: list) -> np.ndarray:
    """Converts string class labels to integer indices."""
    return label_encoder.transform(labels)


def decode_labels(indices) -> np.ndarray:
    """Converts integer indices back to string class labels."""
    return label_encoder.inverse_transform(indices)


def preprocess(images: list, labels: list) -> tuple[np.ndarray, np.ndarray]:
    """Runs the full preprocessing pipeline and returns normalised images and encoded labels."""
    X = normalise_images(images)
    y = encode_labels(labels)
    return X, y