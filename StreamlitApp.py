# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow
import numpy as np
from keras.models import load_model
import os

# Set TensorFlow log level to suppress unnecessary messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set page title
st.title("PNEUMONIA DETECTION")

# Function to load and preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Load the trained model
def load_trained_model():
    model_path = r'D:\Shamanth\GenAI project\PneumoniaDetectionGenAI.h5'
    model = load_model(model_path, compile=False)
    return model

# Define the optimizer with desired parameters
def define_optimizer(model):
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to make prediction
def make_prediction(model, image):
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        return "Pneumonia Detected"
    else:
        return "Normal"

# Add menu bar with "Home", "Example", and "GitHub" options
menu = st.sidebar.selectbox("Menu", ["Home", "Example", "GitHub"])

# Handle menu selection
if menu == "Home":
    st.write("Welcome to Pneumonia Detection App!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Predict") and uploaded_file is not None:
        st.write("Classifying...")
        img_array = preprocess_image(uploaded_file)
        model = load_trained_model()
        model = define_optimizer(model)
        prediction = make_prediction(model, img_array)
        st.success(f"Prediction: {prediction}")

elif menu == "Example":
    st.write("Sample Output And Difference In Lung X-ray")

    # Add clickable images for classification
    pneumonia_image_path = r"C:\Users\goodb\Downloads\image_25.png"
    normal_image_path = r"C:\Users\goodb\Downloads\image_21.png"

    # Load the trained model
    model = load_trained_model()
    model = define_optimizer(model)

    # Arrange images and captions in columns
    col1, col2 = st.columns(2)

    with col1:
        pneumonia_img = Image.open(pneumonia_image_path)
        st.image(pneumonia_img, use_column_width=True)
        pneumonia_button_key = "pneumonia_button"
        if st.button("Click to Classify", key=pneumonia_button_key):
            img_array = preprocess_image(pneumonia_image_path)
            prediction = make_prediction(model, img_array)
            st.success(f"Prediction: {prediction}")

    with col2:
        normal_img = Image.open(normal_image_path)
        st.image(normal_img, use_column_width=True)
        normal_button_key = "normal_button"
        if st.button("Click to Classify", key=normal_button_key):
            img_array = preprocess_image(normal_image_path)
            prediction = make_prediction(model, img_array)
            st.success(f"Prediction: {prediction}")



elif menu == "GitHub":
    st.title("GitHub Repository")
    st.write("This Streamlit application is for Pneumonia Detection")
    st.write("It includes features like uploading an image for prediction.")
    st.write("Click the button below to visit the GitHub repository.")

    # Add button to redirect to GitHub repository
    github_url = "https://github.com/TSShamanth/Pneumonia-Detection"
    if st.button("Go to GitHub"):
        js = f"window.open('{github_url}', '_blank')"
        html = f"<html><head><script>{js}</script></head><body></body></html>"
        st.components.v1.html(html)
