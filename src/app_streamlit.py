# src/app_streamlit.py
"""
Streamlit app for AI Image Classification demo.
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras


MODEL_PATH = "models/cnn_cifar10.h5"
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
IMG_SIZE = (32, 32)

@st.cache_resource
def load_cnn_model():
    return keras.models.load_model(MODEL_PATH)

model = load_cnn_model()

st.title(" AI-Powered Image Classifier")
st.write("Upload an image to predict its class using the trained CNN model.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img_arr = np.array(image.resize(IMG_SIZE)).astype("float32") / 255.0
    pred = np.argmax(model.predict(np.expand_dims(img_arr, axis=0)), axis=1)[0]
    st.success(f"Predicted Class: **{CLASS_NAMES[pred]}**")
