# 🧠 AI-Powered Image Classification System 

---

## 📋 Overview

This project is an **AI-based image classification system** built using **Convolutional Neural Networks (CNNs)** in **TensorFlow/Keras**. The model is trained on the **CIFAR-10 dataset** to classify images into 10 categories.

---

## 🏷️ Dataset – CIFAR-10

**Total Images:** 60,000 (32×32 RGB)

**Classes:**
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

**Data Split:**
- Train: 70%
- Validation: 15%
- Test: 15%

---

## ⚙️ Features

- End-to-end data pipeline with preprocessing and augmentation
- Custom CNN with 3 convolutional blocks
- Batch Normalization and Dropout for stable training
- Early Stopping and Learning Rate Scheduling
- Performance metrics: Accuracy, Precision, Recall, Confusion Matrix
- Streamlit Web App for real-time image prediction

---

## 🏗️ Project Structure

```
ai_image_classifier/
├─ src/
│ ├─ data_loader.py
│ ├─ model.py
│ ├─ train.py
│ ├─ utils.py
│ └─ app_streamlit.py
├─ models/
│ └─ cnn_cifar10.h5
├─ requirements.txt
└─ README.md
```

---

## 🚀 How to Run

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train the Model

```bash
python -m src.train
```

### Run the Streamlit App

```bash
python -m streamlit run src/app_streamlit.py
```

📈 Results

Validation Accuracy: ~79–81%
Test Accuracy: ~78%
Model saved at: models/cnn_cifar10.h5