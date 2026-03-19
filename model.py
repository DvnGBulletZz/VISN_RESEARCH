# model.py
# Defines the CNN for grid-based multi-object detection.
# Input : full board image (224x224x3)
# Output: (GRID_S, GRID_S, 5 + NUM_CLASSES) — one prediction per grid cell

from tensorflow.keras import layers, models
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, GRID_S


def build_model() -> models.Sequential:
    """Builds and returns the detection CNN."""

    model = models.Sequential([

        # Block 1 — learns basic edges and shapes
        # 224x224 -> 112x112
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        layers.MaxPooling2D((2, 2)),

        # Block 2 — learns more specific piece features
        # 112x112 -> 56x56
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3 — added back because 224x224 input has more spatial detail
        # to extract before the dense layers compared to the old 64x64 patches
        # 56x56 -> 28x28
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Detection head
        # Flatten and project to the grid output shape
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),

        # Output: one vector per grid cell with box coords + confidence + class scores
        layers.Dense(GRID_S * GRID_S * (5 + NUM_CLASSES), activation='sigmoid'),

        # Reshape into the grid tensor (GRID_S, GRID_S, 5 + NUM_CLASSES)
        layers.Reshape((GRID_S, GRID_S, 5 + NUM_CLASSES)),
    ])

    return model
