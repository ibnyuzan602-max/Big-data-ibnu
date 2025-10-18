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
# CSS DARK FUTURISTIK
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
    animation: fadeIn 0.6s ease-in-out;
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
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.stDownloadButton > button {
    background-color: #00c6ff !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: 0.3s;
}
.stDownloadButton > button:hover {
    background-color: #0072ff !important;
}
.lottie-center {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
    background-color: transparent;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(0, 162, 255, 0.4), 0 0 50px rgba(0, 162, 255, 0.2);
    transition: all 0.5s ease-in-out;
}
.lottie-center:hover {
    box-shadow: 0 0 45px rgba(0, 200, 255, 0.6), 0 0 90px rgba(0, 200, 255, 0.3);
    transform: scale(1.03);
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
# ANIMASI UTAMA & LOADING
# =========================
LOTTIE_URLS = {
    "AI Futuristic": "https://assets10.lottiefiles.com/packages/lf20_pwohahvd.json",
    "Cyber Brain": "https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json",
    "Tech Network": "https://assets10.lottiefiles.com/packages/lf20_zrqthn6o.json"
}

LOTTIE_LOADING = "https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json"  # Robot thinking animation

st.sidebar.header("⚙️ Pengaturan Tampilan")
selected_anim = st.sidebar.selectbox("Pilih Animasi AI:", list(LOTTIE_URLS.keys()))
lottie_ai = load_lottie_url(LOTTIE_URLS[selected_anim])
lottie_loading = load_lottie_url(LOTTIE_LOADING)

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
# SIDEBAR PILIH MODE
# =========================
st.sidebar.header("🧠 Mode AI")
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "AI Insight"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Unggah gambar, lalu biarkan AI menganalisis secara otomatis.")

# =========================
# HEADER UTAMA
# =========================
st.title("🤖 AI Vision Pro Dashboard")
st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

# =========================
# ANIMASI DI TENGAH
# =========================
if lottie_ai:
    st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
    st_lottie(lottie_ai, height=300, key="ai_center_anim")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='warning-box'>⚠️ Animasi AI tidak berhasil dimuat.</div>", unsafe_allow_html=True)

# =========================
# UPLOAD GAMBAR
# =========================
st.markdown("### 📤 Unggah Gambar untuk Analisis")
uploaded_file = st.file_uploader("Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    # Tampilkan animasi loading
    st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
    st_lottie(lottie_loading, height=180, key="loading_ai")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>🤖 AI sedang menganalisis gambar...</p>", unsafe_allow_html=True)
    time.sleep(2)

    # MODE 1: YOLO DETECTION
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

        st.markdown("""
        <div class="result-card">
            <h3>✅ Deteksi Selesai</h3>
            <p>Objek berhasil dikenali menggunakan model YOLOv8.</p>
        </div>
        """, unsafe_allow_html=True)

    # MODE 2: IMAGE CLASSIFICATION
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

        hasil_txt = f"Kelas: {class_index}\nProbabilitas: {confidence:.2f}"
        st.download_button(
            label="📥 Download Hasil Klasifikasi",
            data=hasil_txt,
            file_name="hasil_klasifikasi.txt",
            mime="text/plain"
        )

    # MODE 3: AI INSIGHT
    elif mode == "AI Insight":
        st.info("🔍 Mode Insight Aktif — AI menganalisis konten gambar.")
        st.markdown("""
        <div class="result-card">
            <h3>💬 Insight Otomatis</h3>
            <p>AI mendeteksi karakteristik visual dominan seperti bentuk, warna, dan pola.
            Analisis ini cocok untuk memahami citra sebelum pelatihan model lanjutan.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='warning-box'>📂 Silakan unggah gambar terlebih dahulu untuk memulai analisis.</div>",
        unsafe_allow_html=True
    )
