# model.py
# Defines the CNN for grid-based multi-object detection.
# Input : 640x640x3
# Output: (GRID_S, GRID_S, 5 + NUM_CLASSES) — one prediction per grid cell

from tensorflow.keras import layers, Model, Input
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, GRID_S


def _cbl(x, filters, kernel=3):
    """Conv + BatchNorm + LeakyReLU."""
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)


def _res(x, filters):
    """Residual block with skip connection."""
    skip = x
    x = _cbl(x, filters // 2, kernel=1)
    x = _cbl(x, filters, kernel=3)
    return layers.Add()([skip, x])


def build_model() -> Model:
    """Builds and returns the detection CNN."""
    inp = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='image')

    # Block 1 — 640x640 -> 320x320
    x = _cbl(inp, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2 — 320x320 -> 160x160
    x = _cbl(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3 — 160x160 -> 80x80
    x = _cbl(x, 128)
    x = _res(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4 — 80x80 -> 40x40
    x = _cbl(x, 256)
    x = _res(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5 — 40x40 -> 20x20 (matches GRID_S=20)
    x = _cbl(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)

    # Detection head — stays spatial at 20x20, no flattening
    x = _cbl(x, 256, kernel=1)
    x = layers.Dropout(0.3)(x)

    out = layers.Conv2D(5 + NUM_CLASSES, 1, activation='sigmoid',
                        padding='same', name='grid')(x)

    return Model(inp, out, name='ChessDetector')
