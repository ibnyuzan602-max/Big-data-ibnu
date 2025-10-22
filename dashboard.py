import streamlit as st
from streamlit_lottie import st_lottie
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import io
import time
import os

# ========== KONFIGURASI DASAR ==========
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# ========== CSS FUTURISTIK ==========
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
    font-family: 'Poppins', sans-serif;
}
button {
    border-radius: 12px !important;
}
.main-title {
    text-align: center;
    font-size: 3em;
    font-weight: 600;
    margin-top: 1.5em;
    color: #00c6ff;
}
.sub-title {
    text-align: center;
    font-size: 1.2em;
    color: #aaa;
}
.center-btn {
    display: flex;
    justify-content: center;
    gap: 1em;
    margin-top: 2em;
}
</style>
""", unsafe_allow_html=True)

# ========== FUNGSI LOAD ANIMASI ==========
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ========== ANIMASI ==========
LOTTIE_WELCOME = "https://lottie.host/8e5c3c3b-6e32-46b7-823b-bbda47fd5b3f/lottie.json"
LOTTIE_DASHBOARD = "https://lottie.host/41e32f9c-bd7f-4828-b76b-0b362c4b1c5a/lottie.json"
LOTTIE_TRANSITION = "https://lottie.host/7a0c4bfa-27e5-4f7a-a158-75a9e4ec51a3/lottie.json"  # loading anim

# ========== LOAD MODEL ==========
@st.cache_resource
def load_models():
    yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
    classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ========== HALAMAN WELCOME ==========
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    st.markdown("<h1 class='main-title'>🤖 Selamat Datang di AI Vision Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</p>", unsafe_allow_html=True)
    
    lottie = load_lottie_url(LOTTIE_WELCOME)
    if lottie:
        st_lottie(lottie, height=300, key="welcome_anim")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.container():
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🚀 Masuk ke Website", use_container_width=True):
                    st.session_state.page = "transition"
            with col_btn2:
                st.button("❌ Tidak", use_container_width=True)

# ========== HALAMAN TRANSISI ==========
elif st.session_state.page == "transition":
    st_lottie(load_lottie_url(LOTTIE_TRANSITION), height=250, key="transition_anim")
    st.markdown("<p style='text-align:center;'>⚙️ Sedang memuat dashboard...</p>", unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.page = "dashboard"
    st.rerun()

# ========== HALAMAN DASHBOARD ==========
elif st.session_state.page == "dashboard":
    st.markdown("<h1 class='main-title'>AI Vision Pro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Deteksi dan Klasifikasi Gambar Cerdas</p>", unsafe_allow_html=True)

    lottie = load_lottie_url(LOTTIE_DASHBOARD)
    if lottie:
        st_lottie(lottie, height=250, key="dashboard_anim")

    st.sidebar.header("🧠 Pilih Mode AI")
    mode = st.sidebar.radio("Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)
        with st.spinner("Model sedang menganalisis gambar..."):
            time.sleep(1.5)

        # YOLO
        if mode == "Deteksi Objek (YOLO)":
            img_cv2 = np.array(img)
            results = yolo_model.predict(source=img_cv2)
            result_img = results[0].plot()
            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

        # KLASIFIKASI
        else:
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            st.success(f"✅ Hasil: Kelas {class_index} ({confidence:.2%})")

    st.markdown("---")
    if st.button("🔙 Kembali ke Halaman Awal"):
        st.session_state.page = "welcome"
        st.rerun()
