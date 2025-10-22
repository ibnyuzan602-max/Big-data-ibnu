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
    initial_sidebar_state="expanded",
)

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
LOTTIE_WELCOME = "https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json"  # animasi sambutan
LOTTIE_MAIN = "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json"   # animasi dashboard
LOTTIE_LOADING = "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json"

# =========================
# CSS & ANIMASI HALUS
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
    transition: opacity 1s ease-in-out;
    opacity: 0;
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
}
.progress-bar {
    width: 100%;
    height: 22px;
    border-radius: 10px;
    overflow: hidden;
    background: #444;
    margin-top: 10px;
}
.progress-fill {
    height: 100%;
    text-align: center;
    color: white;
    font-weight: bold;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
    background-color: transparent;
    border-radius: 20px;
    padding: 20px;
}
.warning-box {
    background-color: rgba(255, 193, 7, 0.1);
    border-left: 5px solid #ffc107;
    color: #ffc107;
    padding: 10px;
    border-radius: 8px;
    margin-top: 15px;
    text-align: center;
    width: 90%;
    margin-left: auto;
    margin-right: auto;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {
    const container = document.querySelector('[data-testid="stAppViewContainer"]');
    if (container) {
        setTimeout(() => { container.style.opacity = 1; }, 150);
    }
});
</script>
""", unsafe_allow_html=True)

# =========================
# STATE HALAMAN
# =========================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# =========================
# FUNGSI NAVIGASI
# =========================
def go_to_dashboard():
    st.session_state.page = "dashboard"

def go_back_home():
    st.session_state.page = "welcome"

# =========================
# HALAMAN 1: WELCOME
# =========================
if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>🤖 Selamat Datang di <span style='color:#00c6ff;'>AI Vision Pro</span></h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)
    
    lottie_welcome = load_lottie_url(LOTTIE_WELCOME)
    if lottie_welcome:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_welcome, height=300, key="welcome_anim")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🚀 Masuk ke Website", use_container_width=True):
            with st.spinner("Memuat halaman utama..."):
                time.sleep(1.2)
                go_to_dashboard()
        st.button("❌ Tidak, Keluar", use_container_width=True)

# =========================
# HALAMAN 2: DASHBOARD
# =========================
elif st.session_state.page == "dashboard":
    st.sidebar.title("🧠 Mode AI")
    mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.button("⬅️ Kembali ke Halaman Awal", on_click=go_back_home, use_container_width=True)

    st.title("🤖 AI Vision Pro Dashboard")
    st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

    lottie_main = load_lottie_url(LOTTIE_MAIN)
    if lottie_main:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_main, height=250, key="main_anim")
        st.markdown("</div>", unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        yolo_model = YOLO(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 4.pt"))
        classifier = tf.keras.models.load_model(os.path.join("model", "Ibnu Hawari Yuzan_Laporan 2.h5"))
        return yolo_model, classifier

    yolo_model, classifier = load_models()

    st.markdown("### 📤 Unggah Gambar untuk Analisis")
    uploaded_file = st.file_uploader("Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        lottie_loading = load_lottie_url(LOTTIE_LOADING)
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_loading, height=150, key="loading_anim")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🤖 AI sedang menganalisis gambar...</p>", unsafe_allow_html=True)
        time.sleep(2)

        if mode == "Deteksi Objek (YOLO)":
            st.info("🚀 Menjalankan deteksi objek...")
            img_cv2 = np.array(img)
            results = yolo_model.predict(source=img_cv2)
            result_img = results[0].plot()

            st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

            img_bytes = io.BytesIO()
            Image.fromarray(result_img).save(img_bytes, format="PNG")
            img_bytes.seek(0)

            st.download_button(
                label="📥 Download Hasil Deteksi",
                data=img_bytes,
                file_name="hasil_deteksi_yolo.png",
                mime="image/png"
            )

        elif mode == "Klasifikasi Gambar":
            st.info("🧠 Menjalankan klasifikasi gambar...")
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
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{confidence*100}%;">{confidence:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='warning-box'>📂 Silakan unggah gambar terlebih dahulu untuk memulai analisis.</div>",
            unsafe_allow_html=True
        )
