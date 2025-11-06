# src/train.py
"""
train.py
Main training pipeline: loads data, builds model, trains and evaluates.
"""

import os
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from src.data_loader import load_dataset, partition_data, get_augmentation
from src.model import build_cnn_model
from src.utils import plot_history, evaluate_and_report, show_samples

# Configuration
DATASET = "cifar10"
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 30
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"cnn_{DATASET}.h5")

def main():
    # Load and split data
    x_train_all, y_train_all, x_test, y_test, class_names = load_dataset(DATASET, IMG_SIZE)
    x_train, y_train, x_val, y_val, x_test, y_test = partition_data(x_train_all, y_train_all)

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    aug = get_augmentation()
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))

    # Build model
    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=len(class_names))
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    # Train
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=cb)

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # Visuals
    plot_history(history)
    show_samples(x_test, y_test, class_names)

    # Evaluate
    evaluate_and_report(model, x_test, y_test, class_names)

if __name__ == "__main__":
    main()
