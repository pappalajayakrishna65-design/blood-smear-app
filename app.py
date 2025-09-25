
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

model = YOLO('best.pt')

st.title("AI Blood Smear Detection Web App")
st.write("Upload a blood smear image to detect cell types.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting...")

    results = model.predict(np.array(image))

    result_img = results[0].plot()
    result_img_rgb = result_img[..., ::-1]

    st.image(result_img_rgb, caption='Detection Result.', use_column_width=True)
