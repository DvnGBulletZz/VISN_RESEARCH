# train.py
# Traint de classifier en slaat model + plots op.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import BATCH_SIZE, EPOCHS, PLOTS_DIR, MODELS_DIR, MODEL_SAVE_PATH, RUN_ID


def train(model, X_train, y_train, X_val, y_val):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"[train] Model opgeslagen → {MODEL_SAVE_PATH}")

    _plot_training(history)
    return history


def _plot_training(history):
    epochs = list(range(1, len(history.history['loss']) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    df_loss = pd.DataFrame({
        'epoch': epochs * 2,
        'loss':  history.history['loss'] + history.history['val_loss'],
        'split': ['train'] * len(epochs) + ['validation'] * len(epochs),
    })
    sns.lineplot(data=df_loss, x='epoch', y='loss', hue='split', ax=ax1)
    ax1.set_title(f"Loss (run {RUN_ID})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

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
    print(f"[train] Trainingscurve opgeslagen → {save_path}")
