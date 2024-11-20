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

# Sidebar
with st.sidebar:
    # Title
    st.title("Hai! Nais to see you here <3\nThis dashboard is designed to help you recognize Braille patterns :)")

    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:  
        st.write("")  
        st.image("https://raw.githubusercontent.com/siskakhnnisa/just_gambar/refs/heads/main/person.png", width=150)  # Menampilkan gambar
        st.write("") 

st.header("Braille Pattern Detection Dashboard")  
st.write("Upload a Braille pattern image, and we'll help you identify the letter!")

# Display example images
st.subheader("Example Images")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.image("https://raw.githubusercontent.com/siskakhnnisa/just_gambar/refs/heads/main/pattern_p.png", caption="Example 1", use_column_width=True)

with example_col2:
    st.image("https://raw.githubusercontent.com/siskakhnnisa/just_gambar/refs/heads/main/pattern_v.png", caption="Example 2", use_column_width=True)

with example_col3:
    st.image("https://raw.githubusercontent.com/siskakhnnisa/just_gambar/refs/heads/main/pattern_h.png", caption="Example 3", use_column_width=True)


uploaded_file = st.file_uploader("Upload file gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image in the center
    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, centering the image
    with col2:  # Middle column
        st.image(image, caption="Your Uploaded Image", use_column_width=False, width=300)

    with st.spinner("Hang tight, we're processing the image and making a prediction..."):
        img_array = process_image(image)          
        print("Shape before prediction:", img_array.shape)
        
        try:
            prediction = model.predict(img_array)  
            predicted_class = np.argmax(prediction, axis=1)  
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
            detected_char = alphabet[predicted_class[0]]
            st.success(f"The Braille letter detected is: **{detected_char}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(f"Error during prediction: {e}")
