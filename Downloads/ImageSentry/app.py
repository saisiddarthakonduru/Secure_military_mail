import streamlit as st
import numpy as np
from PIL import Image
import time
import os

# --- Configuration ---
MODEL_FILENAME = 'stego_detector.h5'
IMG_SIZE = 128
st.set_page_config(layout="wide", page_title="AI Counter-Service Dashboard")

# --- Path Fix ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILENAME)

# --- Helper Functions ---

@st.cache_resource
def load_ai_model(path):
    """
    Attempts to load the Keras model. 
    Returns the model if found, otherwise returns None to trigger 'Demo Mode'.
    """
    if not os.path.exists(path):
        return None  # Trigger Demo Mode

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Resizes and normalizes the image (simulation or real)."""
    # If using real model, we need cv2. If not, we just pass.
    try:
        import cv2
        img = np.array(image.convert('RGB'))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except ImportError:
        return np.array(image) # Fallback if cv2 isn't installed

def analyze_image(img, model):
    """Scans the image and returns a prediction."""
    start_time = time.time()
    
    # --- REAL MODE ---
    if model is not None:
        processed_img = preprocess_image(img)
        prediction_score = model.predict(processed_img)[0][0]
    
    # --- DEMO / SIMULATION MODE ---
    else:
        # Simulate processing time
        time.sleep(1.5) 
        # Generate a random score for demonstration
        prediction_score = np.random.uniform(0, 1)

    latency = (time.time() - start_time) * 1000  # ms
    is_threat = prediction_score > 0.5
    confidence = prediction_score if is_threat else (1 - prediction_score)
    
    return is_threat, confidence, latency, prediction_score

# --- Main Application UI ---

st.title("🛡️ Secure Email AI: Counter-Service Prototype")

# Check if model exists to decide on the UI message
model = load_ai_model(MODEL_PATH)

if model is None:
    st.warning(f"⚠️ **DEMO MODE ACTIVE**: Could not find '{MODEL_FILENAME}'. Simulating AI results.")
    st.info("To use the real AI, place the 'stego_detector.h5' file in this folder.")
else:
    st.success(f"✅ **REAL MODE ACTIVE**: AI Model '{MODEL_FILENAME}' loaded successfully.")

st.subheader("Upload an image to scan for hidden steganographic threats.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image:")
        img = Image.open(uploaded_file)
        st.image(img, caption=f"'{uploaded_file.name}' - Ready for analysis.", use_column_width=True)

    with col2:
        st.subheader("Analysis Dashboard")
        
        with st.spinner("AI is analyzing the image..."):
            is_threat, confidence, latency, prediction_score = analyze_image(img, model)

        # --- Display Results ---
        if is_threat:
            st.error("### 🔴 THREAT DETECTED")
            st.metric(label="Threat Type", value="Steganography (Hidden Data)")
        else:
            st.success("### 🟢 IMAGE CLEAN")
            st.metric(label="Status", value="No Threat Detected")

        # --- Display M&M Metrics ---
        st.subheader("Measurement & Monitoring (M&M)")
        
        metric1, metric2 = st.columns(2)
        metric1.metric(
            label="Confidence Score", 
            value=f"{confidence * 100:.2f} %"
        )
        metric2.metric(
            label="Analysis Latency (Speed)", 
            value=f"{latency:.0f} ms"
        )

        with st.expander("See Raw Prediction Score"):
            st.write(f"Raw Score: {prediction_score:.4f}")
            st.write("(Score > 0.5 indicates a threat)")