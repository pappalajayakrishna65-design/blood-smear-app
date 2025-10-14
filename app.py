import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Load your trained YOLOv8 model
model = YOLO('best.pt')

st.title("AI Blood Smear Analyzer")
st.write("Upload a blood smear image to detect and count cell types.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("---")
    st.write("### Analyzing...")

    # Predict using YOLO
    results = model.predict(np.array(image))

    # --- NEW: Code to count and display results ---
    if results:
        result = results[0]
        if len(result.boxes) > 0:
            # Count each detected class
            counts = {name: 0 for name in result.names.values()}
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                counts[class_name] += 1

            # Create a DataFrame for a nice table display
            df_counts = pd.DataFrame(list(counts.items()), columns=['Cell Type', 'Count'])

            st.write("#### Detection Summary")
            st.dataframe(df_counts)

            # Draw bounding boxes and display the result image
            result_img = result.plot()
            result_img_rgb = result_img[..., ::-1] # Convert BGR to RGB
            st.image(result_img_rgb, caption='Detection Result', use_container_width=True)
        else:
            st.write("No cells were detected in the image.")
