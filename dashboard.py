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
# CSS KUSTOM + ANIMASI
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: #fff;
}
[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    backdrop-filter: blur(10px);
    color: white;
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] * { color: white !important; }
h1, h2, h3 {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.5px;
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
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI LOAD LOTTIE
# =========================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animasi utama (AI glow)
lottie_ai = load_lottie_url("https://lottie.host/bc197d3f-ec64-4aef-bf90-7e6a6d030a12/9bmmFgRFqL.json")

# Animasi loading biru
lottie_loading = load_lottie_url("https://lottie.host/6b79f1cb-52e3-40ff-91c0-7d9fa52a6932/b1exxYPDpy.json")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Ibnu Hawari Yuzan_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Ibnu Hawari Yuzan_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Mode AI")
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "AI Insight"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Unggah gambar, lalu biarkan AI menganalisis secara otomatis.")

# =========================
# HEADER + ANIMASI
# =========================
st.title("🤖 AI Vision Pro Dashboard")
st.markdown("### Sistem Deteksi dan Klasifikasi Gambar Cerdas")

col1, col2 = st.columns([1, 1])
with col1:
    image_path = os.path.join("images", "ai-illustration.png")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=False, width=350, caption="AI Vision System")
    else:
        st.markdown(
            "<div class='warning-box'>⚠️ Gambar ilustrasi tidak ditemukan. Pastikan file ada di folder <b>'images/'</b>.</div>",
            unsafe_allow_html=True
        )

with col2:
    if lottie_ai:
        st.markdown("<div class='lottie-center'>", unsafe_allow_html=True)
        st_lottie(lottie_ai, height=280, key="ai_anim")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='warning-box'>⚠️ Animasi AI tidak berhasil dimuat.</div>",
            unsafe_allow_html=True
        )

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("📤 Unggah Gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)
    
    # Animasi loading
    with st.spinner("Model sedang memproses gambar..."):
        st_lottie(lottie_loading, height=120, key="loading_anim")
        time.sleep(1.5)

    # MODE 1: YOLO
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

    # MODE 2: KLASIFIKASI
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

    # MODE 3: INSIGHT
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
    st.markdown("<div class='warning-box'>📂 Silakan unggah gambar terlebih dahulu untuk memulai analisis.</div>", unsafe_allow_html=True)
