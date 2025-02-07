import os
os.system('pip install tensorflow')
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load trained custom CNN model
try:
    model = tf.keras.models.load_model("E:\\mohith\\d deta\\aqualife Detection\\venv\\model.h5")  # Path to custom CNN model
except FileNotFoundError:
    st.error("Trained model not found. Ensure the model file is in the specified directory.")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file, target_size):
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Resize image to match model's expected input
        image = image.resize(target_size)
        
        # Convert image to numpy array and normalize pixel values
        image = np.array(image) / 255.0
        
        # If the image is grayscale (only 2D), convert it to RGB (3D)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Add batch dimension (required for prediction)
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit app title
st.title("Image Classification with Custom CNN Model")

# File uploader widget for image input
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Proceed if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Button to trigger prediction
    if st.button("Predict"):
        # Preprocess the uploaded image
        image_input = preprocess_image(uploaded_file, target_size=(64, 64))  # Custom CNN model input size (e.g., 64x64)

        if image_input is not None:
            try:
                # Make predictions with the custom CNN model
                prediction = model.predict(image_input)

                # Get the predicted class index (the class with the highest probability)
                class_index = np.argmax(prediction)

                # Define the list of class labels
                class_labels = [
                    'Angelfish', 'Arowana Fish', 'Catla or Indian Carp fish', 'Betta Fish', 'Discus Fish',
                    'Guppy Fish', 'Clownfish', 'Magur or Walking Catfish fish', 'Hilsa or Ilish Shad fish',
                    'Goldfish', 'Koi Fish', 'Oscar Fish', 'Neon Tetra', 'Tilapia or Cichlid Fish',
                    'Pulasa Fish', 'Rani or Pink Perch fish', 'Tengra or Mystus Tengara fish'
                ]

                # Display the prediction result
                st.markdown("<h1 style='color:blue;'>Custom CNN Model Prediction</h1>", unsafe_allow_html=True)
                st.write(f"Predicted Class: {class_labels[class_index]}", style={'font-size': '20px'})
            except Exception as e:
                st.error(f"Error in prediction: {e}")

