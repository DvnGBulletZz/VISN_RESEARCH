# main.py
# Startpunt voor de chess piece classifier.

from data_loader import load_patches, plot_patch_verification, plot_class_distribution
from model import build_model
from train import train
from evaluate import plot_confusion_matrix
import tensorflow as tf


def setup_gpu(mem_mb=19_456):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)])
        print(f'GPU memory limited to {mem_mb} MB')
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision: mixed_float16')

setup_gpu()

def main():

    print("[main] Data laden...")
    X_train, y_train = load_patches("train")
    X_val,   y_val   = load_patches("valid")
    X_test,  y_test  = load_patches("test")

    print(f"[main] Train: {X_train.shape} | Validatie: {X_val.shape} | Test: {X_test.shape}\n")

    print("[main] Verificatie plots opslaan...")
    plot_patch_verification(X_train, y_train)
    plot_class_distribution(y_train, split="train")

    print("[main] Model bouwen...")
    model = build_model()
    model.summary()

    print("\n[main] Trainen...")
    train(model, X_train, y_train, X_val, y_val)

    print("\n[main] Evalueren op testset...")
    plot_confusion_matrix(model, X_test, y_test)

    print("\n[main] Klaar.")


if __name__ == "__main__":
    main()
