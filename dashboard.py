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

# ===============================
# KONFIGURASI DASAR
# ===============================
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# ===============================
# CSS & ANIMASI TRANSISI (FADE IN/OUT)
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
    opacity: 0;
    transition: opacity 1s ease-in-out;
}
body.loaded [data-testid="stAppViewContainer"] {
    opacity: 1;
}
.fade-out {
    opacity: 0 !important;
    transition: opacity 1s ease-in-out;
}

[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] * { color: white !important; }

h1, h2, h3 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
}

.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
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

.music-bar {
    text-align: center;
    margin-top: 15px;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", function(){
    document.body.classList.add('loaded');
});

function fadeOutAndReload() {
    const container = document.querySelector('[data-testid="stAppViewContainer"]');
    if (container) {
        container.classList.add('fade-out');
        setTimeout(() => { window.location.reload(); }, 900);
    }
}
</script>
""", unsafe_allow_html=True)

# ===============================
# FUNGSI LOTTIE
# ===============================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

# Animasi untuk halaman 1 dan 2
LOTTIE_WELCOME = "https://assets4.lottiefiles.com/packages/lf20_touohxv0.json"  # Robot welcome
LOTTIE_DASHBOARD = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"  # AI analysis
LOTTIE_LOADING = "https://assets7.lottiefiles.com/packages/lf20_j1adxtyb.json"  # smooth loading animation

# ===============================
# SESSION STATE HALAMAN
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to_dashboard():
    st.session_state.page = "dashboard"

def go_back_home():
    st.session_state.page = "welcome"

# ===============================
# PAGE 1: WELCOME
# ===============================
if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>🤖 Selamat Datang di AI Vision Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</p>", unsafe_allow_html=True)

    st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
    st_lottie(load_lottie_url(LOTTIE_WELCOME), height=320, key="welcome_anim")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:30px;'>", unsafe_allow_html=True)
    if st.button("🚀 Masuk ke Website"):
        st.markdown("<script>fadeOutAndReload()</script>", unsafe_allow_html=True)
        with st.spinner("Memuat halaman utama..."):
            st_lottie(load_lottie_url(LOTTIE_LOADING), height=180, key="transition_anim")
            time.sleep(2)
        go_to_dashboard()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Musik latar
    if "show_music" not in st.session_state:
        st.session_state.show_music = True

    if st.session_state.show_music:
        st.markdown("""
        <div class='music-bar'>
            <audio controls autoplay loop>
                <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" type="audio/mpeg">
                Browser kamu tidak mendukung audio.
            </audio>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🎵 Sembunyikan/Tampilkan Musik"):
        st.session_state.show_music = not st.session_state.show_music
        st.rerun()

# ===============================
# PAGE 2: DASHBOARD
# ===============================
elif st.session_state.page == "dashboard":
    st.markdown("<h1 style='text-align:center;'>AI Vision Pro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</p>", unsafe_allow_html=True)

    st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
    st_lottie(load_lottie_url(LOTTIE_DASHBOARD), height=300, key="dash_anim")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.markdown("<script>fadeOutAndReload()</script>", unsafe_allow_html=True)
        go_back_home()
        time.sleep(1)
        st.rerun()

    # ===============================
    # MODEL DAN FITUR AI
    # ===============================
    @st.cache_resource
    def load_models():
        yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
        classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
        return yolo_model, classifier

    yolo_model, classifier = load_models()

    mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        with st.spinner("AI sedang menganalisis..."):
            st_lottie(load_lottie_url(LOTTIE_LOADING), height=150, key="load_ai")
            time.sleep(2)

        if mode == "Deteksi Objek (YOLO)":
            img_cv2 = np.array(img)
            results = yolo_model.predict(source=img_cv2)
            result_img = results[0].plot()
            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

        elif mode == "Klasifikasi Gambar":
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            st.markdown(f"""
            <div style='text-align:center;'>
                <h3>🧾 Hasil Prediksi</h3>
                <p><b>Kelas:</b> {class_index}</p>
                <p><b>Akurasi:</b> {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
