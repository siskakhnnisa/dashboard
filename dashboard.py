import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Fungsi untuk memuat model
@st.cache_resource
def load_braille_model():
    return load_model("braille_model.h5")  # Ganti dengan path model Anda

model = load_braille_model()

# Fungsi untuk memproses gambar
def process_image(image):
    img_size = (28, 28)  # Sesuaikan ukuran input model
    image = ImageOps.grayscale(image)  # Konversi gambar ke grayscale
    image = image.resize(img_size)  # Ubah ukuran gambar
    img_array = np.array(image) / 255.0  # Normalisasi piksel ke rentang [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
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
        predicted_class = np.argmax(prediction, axis=1)  # Ambil kelas dengan probabilitas tertinggi

        # Konversi hasil prediksi ke alfabet
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Mapping kelas ke alfabet
        detected_char = alphabet[predicted_class[0]]

    # Menampilkan hasil prediksi
    st.success(f"Huruf Braille yang terdeteksi: **{detected_char}**")
