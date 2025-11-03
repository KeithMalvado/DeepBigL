# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definisikan parameter yang sama seperti saat training
IMG_SIZE = 224 # [cite: 27, 28]
NUM_CLASSES = 3 # [cite: 30]
MODEL_NAME = 'EfficientNetB6' # [cite: 66]

# --- Fungsi untuk Memuat Model dan Memprediksi ---

@st.cache_resource
def load_model():
    """Memuat model EfficientNetB6 yang telah dilatih."""
    model_path = f'{MODEL_NAME}_best_model.h5'
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

def preprocess_image(image):
    """Memproses gambar agar sesuai dengan input model."""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(model, processed_image):
    """Melakukan prediksi klasifikasi."""
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    class_names = ['Brown Spot', 'Leaf Blight', 'Leaf Smut'] 
    
    predicted_class = class_names[np.argmax(score)]
    confidence = np.max(score) * 100
    return predicted_class, confidence

def main():
    st.title("Klasifikasi Penyakit Daun Padi (EfficientNetB6)")
    st.markdown("Unggah gambar daun padi untuk mendeteksi penyakit.")

    if not os.path.exists(f'{MODEL_NAME}_best_model.h5'):
        st.warning(" File model **EfficientNetB6_best_model.h5** tidak ditemukan. Pastikan Anda menyertakannya di GitHub.")
        return

    model = load_model()

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
        st.write("")
        st.write("Memprediksi...")

        processed_img = preprocess_image(image)
        predicted_class, confidence = predict_class(model, processed_img)

        st.success(f"**Hasil Prediksi:** {predicted_class}")
        st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")

main()