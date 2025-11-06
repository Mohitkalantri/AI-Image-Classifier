"""
data_loader.py — Dataset loading and preprocessing for CIFAR-10
Author: Mohit Kalantri
Organization: Flikt Technology Web Solutions
Description:
Loads CIFAR-10 dataset and splits it into training, validation, and test sets.
Performs normalization and returns processed NumPy arrays.
"""

import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def load_data():
    """Loads and preprocesses CIFAR-10 dataset."""
    
    # Load CIFAR-10 dataset
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to range [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Split training data into train and validation sets (85% train / 15% val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.15, random_state=42
    )

    # Flatten labels
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    print(f"✅ Dataset Loaded Successfully:")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Testing samples: {len(x_test)}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
