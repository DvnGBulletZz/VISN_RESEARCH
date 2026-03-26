# model.py
# Simpel CNN voor classificatie van individuele schaakstukken.

from tensorflow import keras
from config import PATCH_SIZE, NUM_CLASSES


def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(PATCH_SIZE, PATCH_SIZE, 3)),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model
