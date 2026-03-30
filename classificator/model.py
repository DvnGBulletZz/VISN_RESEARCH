# model.py
# Simpel CNN voor classificatie van individuele schaakstukken.

from tensorflow import keras
from config import PATCH_SIZE, NUM_CLASSES


# --- Old custom CNN (run 15-18) ---
def build_model() -> keras.Model:
    lrelu = lambda: keras.layers.LeakyReLU(0.1)
    model = keras.Sequential([
        # 80x80 -> 40x40
        keras.layers.Conv2D(32, (3, 3), use_bias=False,
                            input_shape=(PATCH_SIZE, PATCH_SIZE, 3)),
        
        lrelu(),
        keras.layers.MaxPooling2D(),
        # 40x40 -> 20x20
        keras.layers.Conv2D(64, (3, 3), use_bias=False),
        lrelu(),
        keras.layers.MaxPooling2D(),
        # 20x20 -> 10x10
        keras.layers.Conv2D(128, (3, 3), use_bias=False),
        lrelu(),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256),
        lrelu(),
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model