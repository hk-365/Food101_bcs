import subprocess

subprocess.run(["pip", "install", "-r", "requirements_food101.txt"])

subprocess.run(["git", "add", "requirements.txt", "README.md"])
subprocess.run(["git", "commit", "-m", "Added OpenCV dependency"])
subprocess.run(["git", "push"])


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load the trained model
model = load_model('Food101_model.h5')

# Class labels (adjust based on your Food101 dataset)
Food_list = {
    0: "burger",
    1: "butter_naan",
    2: "chai",
    3: "chapati",
    4: "chole_bhature",
    5: "dal_makhani",
    6: "dhokla",
    7: "fried_rice",
    8: "idli",
    9: "jalebi",
    10: "kathi_roll",
    11: "kadhai_paneer",
    12: "kulfi",
    13: "masala_dosa",
    14: "momos",
    15: "paani_puri",
    16: "pakode",
    17: "pav_bhaji",
    18: "pizza",
    19: "samosa"
}

# Define the prediction function
def predict_food(image):
    # Resize image using Pillow to match model input size
    image = image.resize((128, 128))
    img = np.array(image)  # Convert the PIL image to a NumPy array
    img = preprocess_input(img)  # Preprocess the input (assumes your model's preprocess_input function)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    prediction = model.predict(img)  # Make predictions
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index with the highest score
    confidence = np.max(prediction)  # Get the confidence score
    return Food_list[predicted_class], confidence

# Streamlit App
st.title("Food101 Classifier")
st.write("Upload an image of food, and the model will predict its class!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    
    # Make prediction
    food_class, confidence = predict_food(image)
    
    # Display prediction
    st.write(f"### Predicted Class: **{food_class}**")
    st.write(f"### Confidence: **{confidence * 100:.2f}%**")


