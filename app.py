import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

# Load the trained model (Ensure "model.h5" exists in the same folder)
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found! Please check the file location.")
    st.stop()

try:
    model = keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Function to make predictions
def predict(features):
    features = np.array(features).reshape(1, -1)  # Reshape for model input
    try:
        prediction = model.predict(features)
        return prediction
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None

# Streamlit UI
st.title("🚀 Deep Learning Model Deployment")
st.write("Upload input data or manually enter values to get predictions.")

# Creating input fields for 41 features (change if your model expects a different number)
num_features = 41
user_inputs = []
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0, format="%.5f")
    user_inputs.append(value)

# Prediction button
if st.button("🔍 Predict"):
    if model:
        prediction = predict(user_inputs)
        if prediction is not None:
            st.success(f"🧠 Model Prediction: {prediction}")
    else:
        st.error("⚠️ Model not loaded. Check if 'model.h5' exists!")
