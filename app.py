import streamlit as st
import numpy as np
from PIL import Image
from utils import preprocess_image, predict_class
from tensorflow.keras.models import load_model
from metrics import cnn_metrics, vgg16_metrics
import os

st.set_page_config(page_title="Wildfire Detector", layout="centered")
st.title("Wildfire Image Classification")
st.markdown("Upload a satellite or forest image to check for wildfire presence.")

@st.cache_resource
def load_cached_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    return model

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model_option = st.radio("Select Model", ["Custom CNN", "VGG16"])

    if st.button("Predict"):
        model_path = "resaved_cnn_model.keras" if model_option == "Custom CNN" else "resaved_vgg16_model.h5"

        try:
            model = load_cached_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

        try:
            img_array = preprocess_image(image, model_option)
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            st.stop()

        try:
            prediction, probability = predict_class(model, img_array)
            st.subheader(f"Prediction: **{prediction}**")
            st.write(f"Confidence: `{probability:.2f}`")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

        st.subheader("📊 Model Evaluation Metrics")

        m = cnn_metrics if model_option == "Custom CNN" else vgg16_metrics

        st.markdown(f"**Test Accuracy:** `{m['Accuracy']}%`")
        st.markdown(f"**Test Loss:** `{m['Loss']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Precision**")
            st.json(m["Precision"])
        with col2:
            st.markdown("**Recall**")
            st.json(m["Recall"])

        st.markdown("**F1-Score**")
        st.json(m["F1-Score"])
