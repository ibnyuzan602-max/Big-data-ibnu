import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import time
import io
import os
from streamlit_lottie import st_lottie

# =========================
# KONFIGURASI DASAR
# =========================
st.set_page_config(
    page_title="AI Vision Pro",
    page_icon="🤖",
    layout="wide",
)

# =========================
# CSS FUTURISTIK
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
    transition: opacity 1s ease-in-out;
}
.fade-out {
    opacity: 0;
}
.fade-in {
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
}
.main-button {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 30px;
}
.stButton>button {
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: bold !important;
    border: none !important;
    color: white !important;
    background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
    transition: all 0.3s ease-in-out !important;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0, 198, 255, 0.6);
}
.warning-box {
    background-color: rgba(255, 193, 7, 0.1);
    border-left: 5px solid #ffc107;
    color: #ffc107;
    padding: 10px;
    border-radius: 8px;
    margin-top: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI LOAD LOTTIE
# =========================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

# Animasi
LOTTIE_WELCOME = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"
LOTTIE_LOADING = "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json"
LOTTIE_SUCCESS = "https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json"
LOTTIE_ENTER = "https://assets10.lottiefiles.com/packages/lf20_49rdyysj.json"  # Animasi masuk

lottie_welcome = load_lottie_url(LOTTIE_WELCOME)
lottie_loading = load_lottie_url(LOTTIE_LOADING)
lottie_success = load_lottie_url(LOTTIE_SUCCESS)
lottie_enter = load_lottie_url(LOTTIE_ENTER)

# =========================
# SESSION STATE UNTUK HALAMAN
# =========================
if "page" not in st.session_state:
    st.session_state.page = "landing"

# =========================
# HALAMAN 1: LANDING PAGE
# =========================
if st.session_state.page == "landing":
    # Judul
    st.markdown("<h1 class='fade-in'>🤖 Selamat Datang di <span style='color:#00c6ff;'>AI Vision Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<h3>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)

    # Musik Latar Futuristik
    audio_url = "https://cdn.pixabay.com/download/audio/2023/04/09/audio_1d2f9e7b7d.mp3?filename=future-vision-ambient-146074.mp3"
    st.audio(audio_url, format="audio/mp3", start_time=0)

    # Animasi
    if lottie_welcome:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_welcome, height=300, key="welcome_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tombol Aksi
    st.markdown("<div class='main-button'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Masuk ke Website"):
            st.session_state.page = "transition"
            st.rerun()
    with col2:
        if st.button("❌ Tidak / Keluar"):
            st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# HALAMAN TRANSISI MASUK
# =========================
elif st.session_state.page == "transition":
    st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
    st_lottie(lottie_enter, height=400, key="enter_anim")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>🔄 Memuat Sistem AI Vision Pro...</h3>", unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.page = "main"
    st.rerun()

# =========================
# HALAMAN 2: MAIN DASHBOARD
# =========================
elif st.session_state.page == "main":
    @st.cache_resource
    def load_models():
        yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
        classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
        return yolo_model, classifier

    yolo_model, classifier = load_models()

    st.sidebar.header("🧠 Mode AI")
    mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state.page = "landing"
        st.rerun()

    # Header
    st.markdown("<h1 class='fade-in'>🤖 AI Vision Pro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        # Animasi Loading
        st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
        st_lottie(lottie_loading, height=180, key="loading_ai")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🤖 AI sedang menganalisis gambar...</p>", unsafe_allow_html=True)
        st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            st.progress(i + 1)

        if mode == "Deteksi Objek (YOLO)":
            img_cv2 = np.array(img)
            results = yolo_model.predict(source=img_cv2)
            result_img = results[0].plot()

            st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
            st_lottie(lottie_success, height=150, key="success_yolo")
            st.markdown("</div>", unsafe_allow_html=True)

            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

        elif mode == "Klasifikasi Gambar":
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
            st_lottie(lottie_success, height=150, key="success_class")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card fade-in">
                <h3>🧾 Hasil Klasifikasi</h3>
                <p><b>Kelas:</b> {class_index}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{confidence*100}%;">{confidence:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box fade-in'>📂 Silakan unggah gambar untuk mulai analisis.</div>", unsafe_allow_html=True)
