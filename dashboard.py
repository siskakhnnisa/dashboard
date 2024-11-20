import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_braille_model():
    return load_model("braille_model.h5")  

model = load_braille_model()
def process_image(image, img_height=64, img_width=64):
    image = ImageOps.grayscale(image)  
    image = image.resize((img_width, img_height)) 
    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)     
    return img_array

st.title("Dashboard Deteksi Pola Braille")
st.write("Unggah gambar pola Braille untuk mendeteksi huruf.")

uploaded_file = st.file_uploader("Upload file gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=False,width=400)

    with st.spinner("Memproses gambar dan melakukan prediksi..."):
        img_array = process_image(image)          
        print("Shape before prediction:", img_array.shape)
        
        try:
            prediction = model.predict(img_array)  
            predicted_class = np.argmax(prediction, axis=1)  
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
            detected_char = alphabet[predicted_class[0]]
            st.success(f"Huruf Braille yang terdeteksi: **{detected_char}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(f"Error during prediction: {e}")
