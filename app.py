import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

@st.cache_resource
def load_model():
    model_url = "https://catsvsdogskaran.s3.eu-north-1.amazonaws.com/cvdmodel.h5"
    model_path = "cvdmodel.h5"

    if not os.path.exists(model_path):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
        else:
            st.error("Failed to download model. Please check the URL.")

    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

def preprocess_image(image):
    image = image.convert('RGB')  
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

st.title("Cat 😺vs Dog🐶 Classifier")

# File uploader for user input
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(preprocessed_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if prediction[0][0] <= 0.5:
        result = "Cat "
        gif_path = "templates/kitten-meow.gif" 
    else:
        result = "Dog "
        gif_path = "templates/dog_bu.gif"  

    # Create two columns: one for the GIF and one for the result
    col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

    # Display the GIF in the first column
    with col1:
        st.image(gif_path)

    with col2:
        st.markdown(f"<h1><b> I'm {result}</b></h1>", unsafe_allow_html=True)  
        st.write(f"Prediction: {result} ")
