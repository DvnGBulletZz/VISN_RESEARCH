# main.py
# Entry point — keeps as thin as possible.

from data_loader import load_all_annotations, load_images, plot_class_distribution, plot_bbox_verification
from preprocessing import preprocess
from model import build_model
from train import train
from evaluate import plot_confusion_matrix, plot_mae, plot_map
from predict import plot_predictions
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

setup_gpu()  # must run before any other TF/Keras imports

def main():
    df_train = load_all_annotations(split="train")
    df_val   = load_all_annotations(split="valid")
    df_test  = load_all_annotations(split="test")
 
    plot_class_distribution(df_train, split="train")
 
    X_train, train_ann = load_images(df_train)
    X_val,   val_ann   = load_images(df_val)
    X_test,  test_ann  = load_images(df_test)
 
    plot_bbox_verification(X_train, train_ann)
 
    X_train, _ = preprocess(X_train, train_ann)
    X_val,   _ = preprocess(X_val,   val_ann)
    X_test,  _ = preprocess(X_test,  test_ann)
 
    model = build_model()
    model, _ = train(model, X_train, train_ann, X_val, val_ann)
 
    plot_confusion_matrix(model, X_test, test_ann)
    plot_mae(model, X_test, test_ann)
    plot_map(model, X_test, test_ann)
    plot_predictions(model, X_test, n=2)
 
 
if __name__ == "__main__":
    main()