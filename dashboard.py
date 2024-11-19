import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Fungsi untuk memuat model
@st.cache_resource
def load_braille_model():
    return load_model("braille_model.h5") 

model = load_braille_model()

# Fungsi untuk memproses gambar
def process_image(image):
    img_size = (28, 28) 
    image = ImageOps.grayscale(image)  
    image = image.resize(img_size) 
    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# Judul aplikasi
st.title("Dashboard Deteksi Pola Braille")
st.write("Unggah gambar pola Braille untuk mendeteksi huruf.")

# Input gambar dari user
uploaded_file = st.file_uploader("Upload file gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Memproses gambar dan melakukan prediksi
    with st.spinner("Memproses gambar dan melakukan prediksi..."):
        img_array = process_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)  

        # Konversi hasil prediksi ke alfabet
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
        detected_char = alphabet[predicted_class[0]]

    # Menampilkan hasil prediksi
    st.success(f"Huruf Braille yang terdeteksi: **{detected_char}**")
