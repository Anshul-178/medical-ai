import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Import prediction functions
from pneumonia.model import predict_pneumonia
from tumour.model import predict_tumour
from skin_cancer.model import predict_skin

st.set_page_config(
    page_title="Medical AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical AI")
st.write("Upload a medical image and let AI predict the disease.")

option = st.selectbox(
    "Choose Disease",
    (
        "Pneumonia",
        "Brain Tumour",
        "Skin Cancer"
    )
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            if option == "Pneumonia":
                prediction = predict_pneumonia(image)

            elif option == "Brain Tumour":
                prediction = predict_tumour(image)

            else:
                prediction = predict_skin(image)

        st.success(f"Prediction : {prediction}")