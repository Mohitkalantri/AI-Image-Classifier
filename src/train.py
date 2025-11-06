# src/train.py
"""
train.py
Main training pipeline: loads data, builds model, trains, evaluates, and saves results.
"""

import os
import tensorflow as tf
from keras import callbacks, optimizers
from src.data_loader import load_data  # ✅ Fixed import
from src.model import build_cnn_model
from src.utils import plot_history, evaluate_and_report, show_samples
import numpy as np

# Configuration
DATASET = "cifar10"
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 30
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"cnn_{DATASET}.h5")


def main():
    # ✅ Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # ✅ Build CNN model
    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=len(class_names))
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # ✅ Define Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    # ✅ Train Model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb
    )

    # ✅ Save Training History for Later Evaluation
    np.save(os.path.join(MODEL_DIR, "training_history.npy"), history.history)
    print("✅ Training history saved at models/training_history.npy")

    # ✅ Save Model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # ✅ Visualize Training Results
    plot_history(history)
    show_samples(x_test, y_test, class_names)

    # ✅ Evaluate Model
    evaluate_and_report(model, x_test, y_test, class_names)


if __name__ == "__main__":
    main()
