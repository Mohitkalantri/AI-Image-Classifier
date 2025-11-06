# src/data_loader.py
"""
data_loader.py
Handles loading and preprocessing of datasets for the CNN image classification project.
"""

import numpy as np
import tensorflow as tf
import os
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def load_dataset(dataset_name="cifar10", img_size=(32, 32)):
    """
    Load and preprocess dataset.
    Args:
        dataset_name: 'cifar10', 'fashion_mnist', or 'custom'
        img_size: target image size (tuple)
    Returns:
        x_train, y_train, x_test, y_test, class_names
    """
    if dataset_name.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    elif dataset_name.lower() == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_train = tf.image.resize(x_train, img_size).numpy()
        x_test = tf.image.resize(x_test, img_size).numpy()
        class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

    elif dataset_name.lower() == "custom":
        # Expected folder structure: data/train/, data/val/, data/test/
        train_dir = "data/train"
        test_dir = "data/test"
        val_dir = "data/val"
        if not os.path.exists(train_dir):
            raise FileNotFoundError("Custom dataset folders not found.")
        train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=img_size)
        val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=img_size)
        test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=img_size)
        return train_ds, val_ds, test_ds, train_ds.class_names
    else:
        raise ValueError("Unsupported dataset. Choose cifar10, fashion_mnist, or custom.")

    x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

    return x_train, y_train, x_test, y_test, class_names


def partition_data(x, y, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    """
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    x_test, y_test = x[:test_size], y[:test_size]
    x_val, y_val = x[test_size:test_size+val_size], y[test_size:test_size+val_size]
    x_train, y_train = x[test_size+val_size:], y[test_size+val_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_augmentation():
    """
    Return a data augmentation Sequential model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
    ])
