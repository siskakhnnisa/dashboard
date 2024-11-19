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
def process_image(image, img_height=28, img_width=28):
    image = ImageOps.grayscale(image)  
    image = image.resize((img_width, img_height)) 
    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0) 

     # Debugging: Print the shape of the input image
    print("Image shape:", img_array.shape)
    
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
        img_array = process_image(image)  # Memproses gambar
        
        # Debugging: Print the shape of the image before prediction
        print("Shape before prediction:", img_array.shape)
        
        try:
            prediction = model.predict(img_array)  # Prediksi dengan model
            predicted_class = np.argmax(prediction, axis=1)  # Mengambil indeks kelas dengan probabilitas tertinggi

            # Konversi hasil prediksi ke alfabet (A-Z)
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
            detected_char = alphabet[predicted_class[0]]
            
            # Menampilkan hasil prediksi
            st.success(f"Huruf Braille yang terdeteksi: **{detected_char}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(f"Error during prediction: {e}")

    # Menampilkan hasil prediksi
    st.success(f"Huruf Braille yang terdeteksi: **{detected_char}**")
