import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import time
from streamlit_lottie import st_lottie

# =========================
# KONFIGURASI DASAR
# =========================
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# =========================
# CSS STYLING
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
}
h1, h2, h3 {
    text-align: center !important;
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 15px;
    margin-bottom: 20px;
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

/* Tombol Musik di kanan bawah */
.music-toggle {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    border: none;
    border-radius: 50%;
    width: 55px;
    height: 55px;
    font-size: 22px;
    cursor: pointer;
    box-shadow: 0 0 15px rgba(0, 198, 255, 0.5);
    animation: fadeIn 1.5s ease-in-out;
    transition: transform 0.2s ease-in-out;
    z-index: 999;
}
.music-toggle:hover {
    transform: scale(1.1);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI LOTTIE
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

# =========================
# ANIMASI
# =========================
LOTTIE_WELCOME = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"
LOTTIE_ENTER = "https://assets10.lottiefiles.com/packages/lf20_49rdyysj.json"
LOTTIE_DASHBOARD = "https://assets9.lottiefiles.com/private_files/lf30_editor_jchphu0h.json"

lottie_welcome = load_lottie_url(LOTTIE_WELCOME)
lottie_enter = load_lottie_url(LOTTIE_ENTER)
lottie_dashboard = load_lottie_url(LOTTIE_DASHBOARD)

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "music_on" not in st.session_state:
    st.session_state.music_on = True

# =========================
# MUSIK (autoplay + toggle)
# =========================
MUSIC_URL = "https://cdn.pixabay.com/download/audio/2023/04/09/audio_1d2f9e7b7d.mp3?filename=future-vision-ambient-146074.mp3"

if st.session_state.music_on:
    st.markdown(f"""
    <audio autoplay loop id="bg-music">
        <source src="{MUSIC_URL}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

music_icon = "🔇" if not st.session_state.music_on else "🎵"
music_toggle_html = f"""
<form action="" method="post">
    <button class="music-toggle" name="music_toggle" type="submit">{music_icon}</button>
</form>
"""
st.markdown(music_toggle_html, unsafe_allow_html=True)

if "music_toggle" in st.query_params:
    st.session_state.music_on = not st.session_state.music_on
    st.query_params.clear()
    st.rerun()

# =========================
# HALAMAN LANDING
# =========================
if st.session_state.page == "landing":
    st.markdown("<h1>🤖 Selamat Datang di <span style='color:#00c6ff;'>AI Vision Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<h3>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)

    if lottie_welcome:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_welcome, height=300, key="welcome_anim")
        st.markdown("</div>", unsafe_allow_html=True)

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
# HALAMAN TRANSISI
# =========================
elif st.session_state.page == "transition":
    st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
    st_lottie(lottie_enter, height=400, key="enter_anim")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>🔄 Memuat Sistem AI Vision Pro...</h3>", unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.page = "main"
    st.rerun()

# =========================
# HALAMAN UTAMA
# =========================
elif st.session_state.page == "main":
    st.markdown("<h1>🤖 AI Vision Pro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)

    if lottie_dashboard:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_dashboard, height=250, key="dashboard_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state.page = "landing"
        st.rerun()
