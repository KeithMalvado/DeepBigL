import gdown
import streamlit as st
import tensorflow as tf
import os

# ID file dari Google Drive
file_id = "1aBcDxyzEfgHIJkLMn"
url = f"https://drive.google.com/uc?id={file_id}"
output = "EfficientNetB6_best_model.h5"

# Cek apakah sudah ada file lokal
if not os.path.exists(output):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(url, output, quiet=False)

# Load model
model = tf.keras.models.load_model(output)

st.success("Model berhasil dimuat!")
