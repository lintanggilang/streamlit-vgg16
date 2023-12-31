import streamlit as st
import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image
import cv2

def main():
    # Load the VGG16 model
    model = VGG16(weights='imagenet', include_top=True)

    # Set up the Streamlit app
    st.title('VGG16 Image Classification')
    st.sidebar.info('Created by Lintang Gilang')

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        original_image = Image.open(uploaded_image)  # Save the original uploaded image
        image = np.array(original_image)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # Preprocess the image for VGG16
        image_processed = preprocess_input(np.expand_dims(image, axis=0))

        # Classify the image
        prediction = model.predict(image_processed)

        # Decode the top 5 predictions to get human-readable labels and percentages
        decoded_predictions = decode_predictions(prediction, top=5)
        
        st.subheader("Top 5 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
            st.write(f"{i + 1}: {label} ({score:.2%})")
        
        # Display the original uploaded image
        st.image(original_image)

if __name__ == '__main__':
    main()
