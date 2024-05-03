# Import necessary libraries
import streamlit as st
import tensorflow
from PIL import Image
import numpy as np
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set page title
st.title("PNEUMONIA DETECTION")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

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

model = load_trained_model()

# Define the optimizer with desired parameters
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)

# Compile the model with the defined optimizer
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Function to make prediction
def make_prediction(image):
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        return "Pneumonia Detected"
    else:
        return "Normal"

# Display uploaded image and prediction result
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image and make prediction
    img_array = preprocess_image(uploaded_file)
    prediction = make_prediction(img_array)

    # Display prediction result
    st.success(f"Prediction: {prediction}")

# Add predict button
predict_button = st.button("Predict")

# Handle predict button click
if predict_button and uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file)
    img = img.resize((None,224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Predict the result
    prediction = model.predict(img_array)
    prediction_prob = prediction[0][0]

    # Display the prediction result
    if prediction_prob > 0.5:
        st.write("This is an image of PNEUMONIA")
    else:
        st.write("This is an image of NORMAL LUNG")
