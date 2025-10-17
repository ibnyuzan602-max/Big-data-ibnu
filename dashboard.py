import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests
import cv2

st.set_page_config(page_title="AI Vision App", page_icon="🧠", layout="wide")

# ==========================

# CSS Kustom

# ==========================

st.markdown(""" <style>
body {
background: linear-gradient(135deg, #f9fafc, #eef2f7);
}
.title {
font-size: 2.8rem;
text-align: center;
font-weight: 800;
color: #1e1e2f;
margin-top: 1rem;
}
.subtitle {
text-align: center;
font-size: 1.2rem;
color: #6c757d;
margin-bottom: 2rem;
}
.upload-box {
background-color: white;
padding: 2rem;
border-radius: 20px;
box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
text-align: center;
transition: all 0.3s ease;
}
.upload-box:hover {
transform: scale(1.01);
box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}
.result-card {
background-color: #ffffff;
border-radius: 20px;
box-shadow: 0px 3px 12px rgba(0,0,0,0.1);
padding: 1.5rem;
text-align: center;
margin-top: 1rem;
}
.result-card h3 {
color: #1e1e2f;
margin-bottom: 0.5rem;
}
.result-card p {
font-size: 1rem;
color: #555;
}
.progress {
height: 20px;
border-radius: 10px;
background-color: #e9ecef;
overflow: hidden;
margin-top: 0.5rem;
}
.progress-bar {
height: 100%;
background: linear-gradient(90deg, #00b4d8, #0077b6);
text-align: center;
color: white;
font-weight: bold;
line-height: 20px;
} </style>
""", unsafe_allow_html=True)

# ==========================

# Fungsi Lottie

# ==========================

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie_url("[https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json](https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json)")

# ==========================

# Load Model

# ==========================

@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ibnu Hawari Yuzan_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Ibnu Hawari Yuzan_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================

# Header

# ==========================

st.markdown('<p class="title">🧠 Image Classification & Object Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unggah gambar untuk menganalisis dengan AI pintar</p>', unsafe_allow_html=True)
st_lottie(lottie_ai, height=180, key="ai")

# ==========================

# Sidebar

# ==========================

mode = st.sidebar.selectbox("⚙️ Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# ==========================

# Proses

# ==========================

if uploaded_file:
   img = Image.open(uploaded_file)
   st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

if mode == "Deteksi Objek (YOLO)":
    st.info("🚀 Menjalankan deteksi objek...")
    results = yolo_model(img)
    result_img = results[0].plot()
    st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

    st.markdown("""
    <div class="result-card">
        <h3>✅ Deteksi Selesai</h3>
        <p>Objek berhasil dikenali menggunakan model YOLOv8.</p>
    </div>
    """, unsafe_allow_html=True)

elif mode == "Klasifikasi Gambar":
    st.info("🔍 Menjalankan klasifikasi gambar...")
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"""
    <div class="result-card">
        <h3>🧾 Hasil Prediksi</h3>
        <p><b>Kelas:</b> {class_index}</p>
        <div class="progress">
            <div class="progress-bar" style="width:{confidence*100}%;">{confidence:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
