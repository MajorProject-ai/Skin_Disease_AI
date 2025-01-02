import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(layout="wide")

# Load the trained model
model = tf.keras.models.load_model('your_trained_model.h5') # Replace with your trained model's path

# Define a function to display the sidebar
def display_sidebar():
    st.sidebar.title("Menu")
    st.sidebar.button("Home")
    st.sidebar.button("Demo Video")
    st.sidebar.button("About")
    st.sidebar.button("Contact Us")

# Define a function to display the main content
def display_main_content():
    st.title("Skin Disease Detector")
    st.subheader("Automated Diagnosis of Skin Diseases with Image Recognition")
    st.image("https://i.imgur.com/oM4m52X.png")
    st.subheader("Upload your skin disease image to get a diagnosis")
    st.markdown("Please note that although our model achieves an 92% accuracy rate, its predictions should be considered with a limited guarantee. Determining the precise type of skin lesion should be done by a qualified doctor for an accurate diagnosis.")
    st.checkbox("I understand and accept", key="accept")
    if st.button("Predict"):
        uploaded_file = st.file_uploader("Drag and drop file here", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            # Preprocess the image
            image = image.convert('RGB') # Ensure image has 3 channels (RGB)
            image = image.resize((224, 224)) # Resize to match model input size
            image = np.array(image) / 255.0 # Normalize pixel values
            image = np.expand_dims(image, axis=0) # Add batch dimension

            # Make prediction using the model
            prediction = model.predict(image)

            # Get the class label with the highest probability
            predicted_class = np.argmax(prediction)

            
