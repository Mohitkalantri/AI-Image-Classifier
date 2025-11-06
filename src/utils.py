# src/utils.py
"""
utils.py
Helper functions for evaluation, plotting, and input validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import tensorflow as tf

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i,j])}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, x_test, y_test, class_names):
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Classification Report:\n", classification_report(y_test, preds, target_names=class_names))
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, class_names, False)
    plot_confusion_matrix(cm, class_names, True)

def show_samples(images, labels, class_names, preds=None, n=6):
    plt.figure(figsize=(12,3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        title = class_names[labels[i]]
        if preds is not None:
            title += f"\nPred: {class_names[preds[i]]}"
        plt.title(title)
        plt.axis("off")
    plt.show()
