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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    GRID_S, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH,
    PLOTS_DIR, MODELS_DIR, RUN_ID, RANDOM_SEED,
)
from preprocessing import label_encoder


def encode_targets(annotations: list) -> np.ndarray:
    """
    Encodes annotations into (N, GRID_S, GRID_S, 5 + NUM_CLASSES) target array.
    x, y are stored as absolute normalised image coordinates (0-1).
    w, h are stored as normalised fractions of the image size.
    The cell is determined by which grid cell the box center falls into.
    Using absolute coords (not cell-relative) because with a spatial Conv2D
    model the cell position is already implicit — simpler and more stable.
    """
    targets = np.zeros((len(annotations), GRID_S, GRID_S, 5 + NUM_CLASSES), dtype=np.float32)

    for i, boxes in enumerate(annotations):
        for box in boxes:
            cx = ((box['xmin'] + box['xmax']) / 2) / IMG_WIDTH
            cy = ((box['ymin'] + box['ymax']) / 2) / IMG_HEIGHT
            w  = (box['xmax'] - box['xmin']) / IMG_WIDTH
            h  = (box['ymax'] - box['ymin']) / IMG_HEIGHT

            col = min(int(cx * GRID_S), GRID_S - 1)
            row = min(int(cy * GRID_S), GRID_S - 1)

            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[label_encoder.transform([box['class']])[0]] = 1.0

            # w/h scaled by 7 (original grid size) regardless of GRID_S.
            # With GRID_S=14, w*14 gives targets >1.0 which sigmoid clips —
            # the model can never predict the correct box size. Using 7 keeps
            # targets around 0.9 which sigmoid handles correctly.
            targets[i, row, col] = np.concatenate([
                [cx * GRID_S - col, cy * GRID_S - row,
                 w * 7, h * 7, 1.0],
                one_hot,
            ])

    return targets


def _compute_class_weights(annotations: list) -> tf.Tensor:
    """Computes inverse-frequency class weights as a tensor."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for boxes in annotations:
        for box in boxes:
            counts[label_encoder.transform([box['class']])[0]] += 1
    total   = counts.sum()
    weights = total / (NUM_CLASSES * np.where(counts == 0, 1, counts))
    return tf.constant(weights, dtype=tf.float32)


def make_detection_loss(class_weights: tf.Tensor):
    """Returns a detection loss function with class weights baked in."""
    def detection_loss(y_true, y_pred):
        obj        = y_true[..., 4:5]
        xy_loss    = tf.reduce_mean(obj * tf.square(y_true[..., :2]  - y_pred[..., :2]))
        wh_loss    = 1.0 * tf.reduce_mean(obj * tf.square(y_true[..., 2:4] - y_pred[..., 2:4]))
        coord_loss = xy_loss + wh_loss
        # Confidence loss with obj/noobj weighting — classic YOLO approach.
        # Cells with a piece get weight 5, empty cells get 0.5.
        # Without this the model learns low confidence everywhere because
        # most cells are empty and that minimises average MSE.
        conf_true = y_true[..., 4:5]
        conf_pred = y_pred[..., 4:5]
        conf_loss = tf.reduce_mean(
            10.0 * obj * tf.square(conf_true - conf_pred) +
            0.5 * (1.0 - obj) * tf.square(conf_true - conf_pred)
        )
        cw         = tf.cast(class_weights, y_true.dtype)
        w_per_cell = tf.reduce_sum(y_true[..., 5:] * cw, axis=-1, keepdims=True)
        class_loss = tf.reduce_mean(
            obj * w_per_cell * tf.keras.losses.categorical_crossentropy(
                y_true[..., 5:], y_pred[..., 5:]
            )[..., tf.newaxis]
        )
        return coord_loss + conf_loss + class_loss
    return detection_loss


def train(model, X_train, y_train_ann, X_val, y_val_ann):
    """Compiles, trains and saves the best model based on validation loss."""
    tf.random.set_seed(RANDOM_SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, f"best_model_run{RUN_ID}.h5"),
        monitor='val_loss', save_best_only=True, verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6,
    )

    class_weights = _compute_class_weights(y_train_ann)
    loss_fn = make_detection_loss(class_weights)

    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss=loss_fn, metrics=['accuracy'])

    history = model.fit(
        X_train, encode_targets(y_train_ann),
        validation_data=(X_val, encode_targets(y_val_ann)),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    _plot_training(history)
    return model, history


def _plot_training(history):
    """Saves loss and accuracy plots side by side."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = list(range(1, len(history.history['loss']) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    df_loss = pd.DataFrame({
        'epoch': epochs * 2,
        'loss':  history.history['loss'] + history.history['val_loss'],
        'split': ['train'] * len(epochs) + ['validation'] * len(epochs),
    })
    sns.lineplot(data=df_loss, x='epoch', y='loss', hue='split', ax=ax1)
    ax1.set_title(f"Loss (run {RUN_ID})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    df_acc = pd.DataFrame({
        'epoch':    epochs * 2,
        'accuracy': history.history['accuracy'] + history.history['val_accuracy'],
        'split':    ['train'] * len(epochs) + ['validation'] * len(epochs),
    })
    sns.lineplot(data=df_acc, x='epoch', y='accuracy', hue='split', ax=ax2)
    ax2.set_title(f"Accuracy (run {RUN_ID})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"training_run{RUN_ID}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Training plot saved → {save_path}")