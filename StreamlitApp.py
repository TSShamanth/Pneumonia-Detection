import streamlit as st
import tensorflow as tf
from keras.models import load_model
import requests

# Set TensorFlow log level to suppress unnecessary messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set page title
st.title("PNEUMONIA DETECTION")

# Function to preprocess the image URL
def preprocess_image(image_url):
    img_response = requests.get(image_url)
    img_data = img_response.content
    return img_data

# Load the trained model from GitHub
#@st.cache(allow_output_mutation=True)
def load_trained_model():
    st.write("Loading model...")
    model_path = "https://github.com/Ajay2k4/Pneumonia-Detection/blob/main/PneumoniaDetectionGenAI.h5"
    model_data = requests.get(model_path).content
    with open('PneumoniaDetectionGenAI.h5', 'wb') as f:
        f.write(model_data)
    model = load_model('PneumoniaDetectionGenAI.h5', compile=False)
    st.write("Model loaded successfully.")
    return model

# Define the optimizer with desired parameters
def define_optimizer(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to make prediction
def make_prediction(model, image_data):
    img_array = tf.image.decode_image(image_data, channels=3)
    img_array = tf.image.resize(img_array, [224, 224]) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
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

    # Assign a unique key to the st.button widget
    predict_button_key = "predict_button"

    if st.button("Predict", key=predict_button_key):
        if uploaded_file is not None:
            st.write("Classifying...")
            img_data = uploaded_file.read()
            model = load_trained_model()
            model = define_optimizer(model)
            prediction = make_prediction(model, img_data)
            st.success(f"Prediction: {prediction}")
        else:
            st.error("NO IMAGE FOUND")
elif menu == "Example":
    st.write("Sample Output And Difference In Lung X-ray")

    # Add clickable images for classification
    pneumonia_image_url = "https://github.com/TSShamanth/Pneumonia-Detection/raw/main/PreprocessedImages/image_25.png"
    normal_image_url = "https://github.com/TSShamanth/Pneumonia-Detection/raw/main/PreprocessedImages/image_21.png"

    # Load the trained model
    model = load_trained_model()
    model = define_optimizer(model)

    # Display images and classify on button click
    col1, col2 = st.columns(2)
    with col1:
        st.image(pneumonia_image_url, caption="Pneumonia Image", use_column_width=True)
        if st.button("Classify Pneumonia"):
            img_data = preprocess_image(pneumonia_image_url)
            prediction = make_prediction(model, img_data)
            st.success(f"Prediction: {prediction}")

    with col2:
        st.image(normal_image_url, caption="Normal Image", use_column_width=True)
        if st.button("Classify Normal"):
            img_data = preprocess_image(normal_image_url)
            prediction = make_prediction(model, img_data)
            st.success(f"Prediction: {prediction}")

elif menu == "GitHub":
    st.title("GitHub Repository")
    st.write("This Streamlit application is for Pneumonia Detection.")
    st.write("It includes features like uploading an image for prediction.")
    st.write("Click the button below to visit the GitHub repository.")

    # Add button to redirect to GitHub repository
    github_url = "https://github.com/TSShamanth/Pneumonia-Detection"
    if st.button("Go to GitHub"):
        st.markdown(f"[GitHub Repository]({github_url})")
