import streamlit as st
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Class names for the first model
class_names_model1 = [
    'Tomato blight disease',
    'Bacterial spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus',
    'Target Spot',
    'Powdery mildew',
    'Spider mites Two spotted spider mite'
]

# Class names for the second model
class_names_model2 = [
    'Cercospora',
    'Bacterial Blight',
    'Anthracnose',
    'Alternaria'
]

# Load the first model
model1 = keras.models.load_model('keras_model1.h5')

# Load the second model
model2 = keras.models.load_model('pomegranate_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = img.convert('RGB')  # Convert image to RGB format
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return keras.applications.mobilenet.preprocess_input(img_array)

def main():
    # ... (rest of your code)

    st.subheader("Select the type of leaf for Model 1")
    selected_leaf_type_model1 = st.selectbox("Leaf Type Model 1", class_names_model1)

    st.subheader("Select the type of leaf for Model 2")
    selected_leaf_type_model2 = st.selectbox("Leaf Type Model 2", class_names_model2)

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions for Model 1
        predictions_model1 = model1.predict(processed_image)
        predicted_class_index_model1 = predictions_model1.argmax()
        predicted_class_name_model1 = class_names_model1[predicted_class_index_model1]
        confidence_model1 = predictions_model1[0][predicted_class_index_model1] * 100

        # Make predictions for Model 2
        predictions_model2 = model2.predict(processed_image)
        predicted_class_index_model2 = predictions_model2.argmax()
        predicted_class_name_model2 = class_names_model2[predicted_class_index_model2]
        confidence_model2 = predictions_model2[0][predicted_class_index_model2] * 100

        # Display predicted class and confidence for Model 1
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Predicted Class Model 1</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{}</h3>
            </div>
            """.format(predicted_class_name_model1),
            unsafe_allow_html=True
        )

        # ... (rest of your code for Model 1)

        # Display predicted class and confidence for Model 2
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Predicted Class Model 2</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{}</h3>
            </div>
            """.format(predicted_class_name_model2),
            unsafe_allow_html=True
        )

        # ... (rest of your code for Model 2)

if __name__ == '__main__':
    main()

