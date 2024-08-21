import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from deepface import DeepFace

# Title
st.title("Emotion Detection from Image")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image with PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to numpy array and then to BGR (OpenCV format)
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect emotions using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'])

        # Print the result to understand its structure
        st.write(result)

        # Handle different possible structures
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # DeepFace may return a list of results

        if 'emotion' in result:
            st.write("Detected emotions:")
            for emotion, score in result['emotion'].items():
                st.write(f"{emotion.capitalize()}: {score:.2f}")

            # Highlight the dominant emotion
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
            st.success(f"Dominant Emotion: {dominant_emotion.capitalize()}")
        else:
            st.error("No emotions detected. Please try a different image.")
        
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid image file.")
else:
    st.warning("Please upload an image.")
