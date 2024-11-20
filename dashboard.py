import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Fungsi untuk memuat model
@st.cache_resource
def load_braille_model():
    return load_model("braille_model.h5")  # Load the Braille model

model = load_braille_model()

# Fungsi untuk memproses gambar
# Update image size to be larger
def process_image(image, img_height=64, img_width=64):
    # Konversi gambar ke grayscale
    image = ImageOps.grayscale(image)  
    # Resize gambar agar sesuai dengan ukuran input model
    image = image.resize((img_width, img_height)) 
    img_array = np.array(image) / 255.0  # Normalisasi nilai piksel
    img_array = np.expand_dims(img_array, axis=-1)  # Tambahkan dimensi channel (grayscale)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch (1 gambar)
    
    return img_array

# Judul aplikasi
st.title("Dashboard Deteksi Pola Braille")
st.write("Unggah gambar pola Braille untuk mendeteksi huruf.")

# Tambahkan gaya CSS untuk memusatkan elemen sepenuhnya
st.markdown(
    """
    <style>
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;  /* Memastikan kontainer memenuhi tinggi viewport */
        flex-direction: column;
    }
    .center-image {
        max-width: 300px;
        max-height: 300px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Input gambar dari user
uploaded_file = st.file_uploader("Upload file gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah dengan ukuran lebih kecil dan posisi tengah
    image = Image.open(uploaded_file)
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(image, caption="Gambar yang diunggah", use_column_width=False, width=300)  # Atur lebar gambar
    st.markdown('</div>', unsafe_allow_html=True)

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
