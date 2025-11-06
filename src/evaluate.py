"""
evaluate.py — Model Evaluation Script for AI-Powered Image Classification System
Author: Mohit Kalantri
Organization: Flikt Technology Web Solutions
Description:
This script evaluates the trained CNN model on validation and test datasets and
generates key metrics (Accuracy, Precision, Recall, F1-Score) and visualizations
(Confusion Matrix, Accuracy/Loss Curves).
"""
import matplotlib
matplotlib.use('Agg')  # Prevents GUI errors when saving plots


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from src.data_loader import load_data

# Define class names for CIFAR-10
CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def evaluate_model(model, x_val, y_val, x_test, y_test, history=None):
    """Evaluate model on validation and test datasets."""
    
    print("\n📊 Evaluating Model on Validation Set...")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)
    print(f"Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

    print("\n📊 Evaluating Model on Test Set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    print("\n🔍 Generating Classification Report...")
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Create 'results' folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save plot as image
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved at results/confusion_matrix.png")

    # Plot training curves if history is available
    if history:
        print("\n📈 Plotting Training Curves...")
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/training_curves.png")
        plt.close()
        print("✅ Training accuracy/loss curves saved at results/training_curves.png")

def main():
    print("🚀 Starting Model Evaluation...")

    # Load dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Load the trained model
    model_path = os.path.join("models", "cnn_cifar10.h5")
    if not os.path.exists(model_path):
        print("❌ Model file not found! Please train the model first.")
        return

    model = load_model(model_path)
    print(f"✅ Loaded model from {model_path}")

    # If you saved history in a .npy file earlier, you can optionally load it
    history_path = os.path.join("models", "training_history.npy")
    history = None
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()

    # Run evaluation
    evaluate_model(model, x_val, y_val, x_test, y_test, history)

if __name__ == "__main__":
    main()
