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
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image is not None:
        image = np.array(Image.open(image))
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # Preprocess the image for VGG16
        image_processed = preprocess_input(np.expand_dims(image, axis=0))

        # Classify the image
        prediction = model.predict(image_processed)

        # Decode the prediction to get human-readable label
        decoded_predictions = decode_predictions(prediction, top=1)
        label = decoded_predictions[0][0][1]

        # Display the classification
        st.subheader("Predicted label:")
        st.write('label:', label)
        st.image(image)

if __name__ == '__main__':
    main()
