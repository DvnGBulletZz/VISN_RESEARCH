# model.py
# Defines the CNN for grid-based multi-object detection.
# Input : 224x224x3
# Output: (GRID_S, GRID_S, 5 + NUM_CLASSES) — one prediction per grid cell
#
# Key change from previous version:
# GlobalAveragePooling replaced by a 5th MaxPool + Conv2D output.
# GAP destroyed all spatial information — the model could learn WHAT is on
# the board but not WHERE. Now the feature map is kept spatial all the way
# to the output, so each cell in the grid directly corresponds to a region
# of the input image.

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

    # Block 1 — 224x224 -> 112x112
    x = _cbl(inp, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2 — 112x112 -> 56x56
    x = _cbl(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3 — 56x56 -> 28x28
    x = _cbl(x, 128)
    x = _res(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4 — 28x28 -> 14x14
    x = _cbl(x, 256)
    x = _res(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5 — 14x14 -> 7x7 (= GRID_S)
    # This pool brings the spatial size to exactly the grid size so each
    # feature map cell aligns with one grid cell in the output
    x = _cbl(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)

    # Detection head — stays spatial, no flattening
    x = _cbl(x, 256, kernel=1)  # 1x1 conv to mix channels
    x = layers.Dropout(0.3)(x)

    # Sigmoid on all outputs — keeps everything in (0,1)
    out = layers.Conv2D(5 + NUM_CLASSES, 1, activation='sigmoid',
                        padding='same', name='grid')(x)

    return Model(inp, out, name='ChessDetector')