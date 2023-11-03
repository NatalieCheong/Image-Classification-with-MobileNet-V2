#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

# Function to preprocess the image for the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to 224x224
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Define the Streamlit app
def main():
    st.title("Image Classification with MobileNetV2")
    st.write("Upload an image and the model will classify it.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        image = preprocess_image(image)
        preds = model.predict(image)
        pred_class = decode_predictions(preds, top=1)[0][0][1]
        st.write(f"Prediction: {pred_class}")

if __name__ == "__main__":
    main()

