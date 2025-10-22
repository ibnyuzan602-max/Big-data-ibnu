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
# CSS FUTURISTIK + FADE EFFECT
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
}
[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] * { color: white !important; }

h1, h2, h3 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
.result-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 4px 25px rgba(0,0,0,0.25);
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
    transition: opacity 1s ease-in-out;
}
.fade-out {
    opacity: 0 !important;
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

# =========================
# ANIMASI
# =========================
LOTTIE_MAIN = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"  # Cyber Brain
LOTTIE_LOADING = "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json"
LOTTIE_SUCCESS = "https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json"

lottie_ai = load_lottie_url(LOTTIE_MAIN)
lottie_loading = load_lottie_url(LOTTIE_LOADING)
lottie_success = load_lottie_url(LOTTIE_SUCCESS)

# =========================
# LOAD MODEL YOLO DAN CNN
# =========================
@st.cache_resource
def load_models():
    yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
    classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
    return yolo_model, classifier

yolo_model, classifier = load_models()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🧠 Mode AI")
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.markdown("---")

# =========================
# HEADER
# =========================
st.title("🤖 AI Vision Pro Dashboard")
st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

# =========================
# ANIMASI UTAMA
# =========================
if lottie_ai:
    st.markdown("<div class='lottie-center' id='ai-anim'>", unsafe_allow_html=True)
    st_lottie(lottie_ai, height=300, key="ai_main")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    # LOADING ANIMATION
    with st.container():
        anim_spot = st.empty()
        with anim_spot:
            st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
            st_lottie(lottie_loading, height=180, key="loading_anim")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;'>🤖 AI sedang menganalisis gambar...</p>", unsafe_allow_html=True)
            st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                st.progress(i + 1)

    # Setelah loading, kosongkan animasi
    anim_spot.empty()

    # MODE 1: DETEKSI OBJEK
    if mode == "Deteksi Objek (YOLO)":
        img_cv2 = np.array(img)
        results = yolo_model.predict(source=img_cv2)
        result_img = results[0].plot()

        st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
        st_lottie(lottie_success, height=150, key="success_ai")
        st.markdown("</div>", unsafe_allow_html=True)

        st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

    # MODE 2: KLASIFIKASI
    elif mode == "Klasifikasi Gambar":
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown("<div class='lottie-center fade-in'>", unsafe_allow_html=True)
        st_lottie(lottie_success, height=150, key="success_ai_class")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <h3>🧾 Hasil Klasifikasi</h3>
            <p><b>Kelas:</b> {class_index}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{confidence*100}%;">{confidence:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='warning-box'>📂 Silakan unggah gambar untuk mulai analisis.</div>",
        unsafe_allow_html=True
    )
