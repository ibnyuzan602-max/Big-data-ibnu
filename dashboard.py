import streamlit as st
from streamlit_lottie import st_lottie
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import time
import io

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# ===============================
# CSS UNTUK TEMA DAN TRANSISI
# ===============================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 10% 20%, #0b0b17, #1b1b2a 80%);
    color: white;
}
.fade-container {
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.center {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# FUNGSI LOAD LOTTIE
# ===============================
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie berbeda untuk tiap halaman
LOTTIE_WELCOME = load_lottie("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")  # animasi welcome
LOTTIE_MAIN = load_lottie("https://assets10.lottiefiles.com/packages/lf20_t24tpvcu.json")    # animasi dashboard

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_tf_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

yolo_model = load_yolo_model()
tf_model = load_tf_model()

# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "mode" not in st.session_state:
    st.session_state.mode = "Klasifikasi Gambar"

def go_to_dashboard():
    st.session_state.page = "dashboard"
    st.experimental_rerun()

def go_home():
    st.session_state.page = "welcome"
    st.experimental_rerun()

# ===============================
# HALAMAN 1 — WELCOME
# ===============================
if st.session_state.page == "welcome":
    with st.container():
        st.markdown("<div class='fade-container'>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center;'>🤖 Selamat Datang di <span style='color:#00c6ff;'>AI Vision Pro</span></h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)
        st_lottie(LOTTIE_WELCOME, height=280, key="welcome_anim")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Masuk ke Website", use_container_width=True):
                go_to_dashboard()
            st.button("❌ Tidak, Keluar", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# HALAMAN 2 — DASHBOARD
# ===============================
elif st.session_state.page == "dashboard":
    with st.container():
        st.markdown("<div class='fade-container'>", unsafe_allow_html=True)

        # Sidebar navigasi
        st.sidebar.title("🧠 Mode AI")
        st.sidebar.write("Pilih Mode:")
        st.session_state.mode = st.sidebar.radio(
            "Pilih Mode:",
            ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"],
            label_visibility="collapsed"
        )

        st.sidebar.button("⬅️ Kembali ke Halaman Awal", on_click=go_home, use_container_width=True)

        # Judul tengah
        st.markdown("<h1 style='text-align:center;'>🤖 AI Vision Pro Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Sistem Deteksi dan Klasifikasi Gambar Cerdas</h3>", unsafe_allow_html=True)
        st_lottie(LOTTIE_MAIN, height=180, key="main_anim")

        # Upload gambar
        uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

            with st.spinner("🤖 Menganalisis gambar..."):
                time.sleep(1.5)

                # ===================
                # MODE: KLASIFIKASI
                # ===================
                if st.session_state.mode == "🧩 Klasifikasi Gambar":
                    st.subheader("📊 Hasil Klasifikasi Gambar")
                    img_resized = image.load_img(uploaded_file, target_size=(224, 224))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    preds = tf_model.predict(img_array)
                    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
                    for i, (imagenet_id, label, score) in enumerate(decoded):
                        st.write(f"{i+1}. **{label}** — {score*100:.2f}%")

                # ===================
                # MODE: DETEKSI OBJEK
                # ===================
                else:
                    st.subheader("🎯 Hasil Deteksi Objek (YOLOv8)")
                    img_bytes = uploaded_file.read()
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    results = yolo_model(img_pil)

                    for r in results:
                        annotated = r.plot()
                        st.image(annotated, caption="Hasil Deteksi YOLO", use_container_width=True)

            st.success("✅ Analisis selesai!")

        st.markdown("</div>", unsafe_allow_html=True)
