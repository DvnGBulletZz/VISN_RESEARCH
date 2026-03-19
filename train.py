# train.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    GRID_S, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH,
    PLOTS_DIR, MODELS_DIR, RUN_ID, RANDOM_SEED,
)
from preprocessing import label_encoder


def encode_targets(annotations: list) -> np.ndarray:
    """Converts annotation dicts into a (N, GRID_S, GRID_S, 5 + NUM_CLASSES) target array."""
    targets = np.zeros((len(annotations), GRID_S, GRID_S, 5 + NUM_CLASSES), dtype=np.float32)

    for i, boxes in enumerate(annotations):
        for box in boxes:
            cx  = ((box['xmin'] + box['xmax']) / 2) / IMG_WIDTH
            cy  = ((box['ymin'] + box['ymax']) / 2) / IMG_HEIGHT
            col = min(int(cx * GRID_S), GRID_S - 1)
            row = min(int(cy * GRID_S), GRID_S - 1)

            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[label_encoder.transform([box['class']])[0]] = 1.0

            targets[i, row, col] = np.concatenate([
                [cx * GRID_S - col, cy * GRID_S - row,
                 (box['xmax'] - box['xmin']) / IMG_WIDTH,
                 (box['ymax'] - box['ymin']) / IMG_HEIGHT,
                 1.0],
                one_hot,
            ])

    return targets


def detection_loss(y_true, y_pred):
    """
    Custom loss: MSE for boxes and confidence, categorical CE for classes.
    Confidence uses MSE instead of binary crossentropy because every image
    contains pieces — there are no empty background images in the dataset,
    so the sigmoid behaviour of binary CE adds no value here.
    """
    obj = y_true[..., 4:5]
    coord_loss = tf.reduce_mean(obj * tf.square(y_true[..., :4] - y_pred[..., :4]))
    conf_loss  = tf.reduce_mean(tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))
    class_loss = tf.reduce_mean(obj * tf.keras.losses.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:])[..., tf.newaxis])
    return coord_loss + conf_loss + class_loss


def train(model, X_train, y_train_ann, X_val, y_val_ann):
    """Compiles, trains and saves the best model based on validation loss."""
    tf.random.set_seed(RANDOM_SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, f"best_model_run{RUN_ID}.h5"),
        monitor='val_loss', save_best_only=True, verbose=1,
    )

    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss=detection_loss, metrics=['accuracy'])

    history = model.fit(
        X_train, encode_targets(y_train_ann),
        validation_data=(X_val, encode_targets(y_val_ann)),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1,
    )

    _plot_training(history)
    return model, history


def _plot_training(history):
    """Saves loss and accuracy plots side by side in one image."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = list(range(1, len(history.history['loss']) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    df_loss = pd.DataFrame({
        'epoch': epochs * 2,
        'loss':  history.history['loss'] + history.history['val_loss'],
        'split': ['train'] * len(epochs) + ['validation'] * len(epochs),
    })
    sns.lineplot(data=df_loss, x='epoch', y='loss', hue='split', ax=ax1)
    ax1.set_title(f"Loss (run {RUN_ID})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Accuracy
    df_acc = pd.DataFrame({
        'epoch':    epochs * 2,
        'accuracy': history.history['accuracy'] + history.history['val_accuracy'],
        'split':    ['train'] * len(epochs) + ['validation'] * len(epochs),
    })
    sns.lineplot(data=df_acc, x='epoch', y='accuracy', hue='split', ax=ax2)
    ax2.set_title(f"Accuracy (run {RUN_ID})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"training_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Training plot saved → {save_path}")