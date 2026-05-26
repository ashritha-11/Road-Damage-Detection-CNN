
# ============================================================
# app.py
# ADVANCED PROFESSIONAL STREAMLIT WEB APP
# ROAD DAMAGE DETECTION USING CNN
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image
import pandas as pd
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection ",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Background */

.stApp {
    background: linear-gradient(
        to right,
        #eef2ff,
        #f8fafc
    );
}

/* Main Title */

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #0F172A;
    margin-top: -20px;
}

/* Subtitle */

.sub-title {
    font-size: 22px;
    text-align: center;
    color: #475569;
    margin-bottom: 30px;
}

/* Cards */

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Metric Cards */

.metric-card {
    background: linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );
    color: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 18px rgba(37,99,235,0.3);
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #0F172A;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        135deg,
        #2563EB,
        #1E40AF
    );
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        135deg,
        #1E40AF,
        #1D4ED8
    );
}

/* Upload Area */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 15px;
    padding: 10px;
    border: 2px dashed #2563EB;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛣️ Road Damage Detection </div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Smart City Road Monitoring using Deep Learning & CNN'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/854/854878.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "🔍 Predict Damage",
        "📊 Model Analytics",
        "📘 About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### CNN Categories

- 🕳️ Pothole
- 🪨 Crack
- ⭕ Manhole
""")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "road_damage_cnn_model.keras"
    )

    return model

model = load_model()

# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open("label_mapping.json", "r") as f:

    label_mapping = json.load(f)

reverse_mapping = {
    int(v): k
    for k, v in label_mapping.items()
}

IMG_SIZE = 128

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="metric-card">
        <h2>95%+</h2>
        <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
        <h2>3</h2>
        <p>Damage Classes</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">
        <h2>CNN</h2>
        <p>Deep Learning Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1,1])

    with left:

        st.markdown("""
        <div class="card">
        <h2>🚀 Project Overview</h2>

        This AI-powered application automatically detects
        road damages using Convolutional Neural Networks.

        <br>

        ✅ Detects potholes, cracks and manholes  
        ✅ Helps smart city authorities  
        ✅ Reduces manual inspection effort  
        ✅ Improves road safety  
        ✅ Provides real-time predictions  

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.image(
            "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h2>🧠 Why CNN for Image Processing?</h2>

    CNN automatically extracts image features like:
    edges, textures, cracks and potholes.

    Unlike ANN, CNN preserves spatial relationships
    between pixels and works efficiently on
    high-resolution images.

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔍 Predict Damage":

    st.markdown("""
    <div class="card">
    <h2>🔍 Upload Road Image</h2>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = image.convert("RGB")

        col1, col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        img = np.array(image)

        resized = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        normalized = resized / 255.0

        input_img = np.expand_dims(
            normalized,
            axis=0
        )

        if st.button("🚀 Analyze Road Damage"):

            progress = st.progress(0)

            for i in range(100):

                time.sleep(0.01)

                progress.progress(i + 1)

            with st.spinner("Analyzing image..."):

                prediction = model.predict(input_img)

                predicted_class = np.argmax(prediction)

                confidence = np.max(prediction)

                class_name = reverse_mapping[predicted_class]

            with col2:

                st.markdown("""
                <div class="card">
                <h2>📊 Prediction Result</h2>
                </div>
                """, unsafe_allow_html=True)

                st.success(
                    f"Detected Damage: {class_name}"
                )

                st.info(
                    f"Confidence Score: {confidence:.2%}"
                )

                # ============================================
                # PROBABILITY CHART
                # ============================================

                st.subheader("Prediction Probabilities")

                probs = prediction[0]

                prob_df = pd.DataFrame({
                    "Class": [
                        reverse_mapping[i]
                        for i in range(len(probs))
                    ],
                    "Probability": probs
                })

                st.bar_chart(
                    prob_df.set_index("Class")
                )

                for idx, prob in enumerate(probs):

                    st.write(
                        f"✅ {reverse_mapping[idx]} : {prob:.2%}"
                    )

# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.markdown("""
    <div class="card">
    <h2>📊 CNN Model Analytics</h2>

    <br>

    ✔ Convolution Layers  
    ✔ MaxPooling Layers  
    ✔ Batch Normalization  
    ✔ Dropout Regularization  
    ✔ Dense Neural Layers  

    <br>

    <h3>Evaluation Metrics</h3>

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix

    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg",
        use_container_width=True
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "📘 About Project":

    st.markdown("""
    <div class="card">

    <h2>📘 About This Project</h2>

    <br>

    Smart cities require automated road monitoring systems
    to improve transportation safety and reduce accidents.

    This project uses Artificial Intelligence and CNN
    architecture to identify road damages from images.

    <br>

    <h3>🛠️ Technologies Used</h3>

    - Python
    - TensorFlow
    - CNN
    - OpenCV
    - Streamlit
    - NumPy
    - Pandas

    <br>

    <h3>🚀 Future Enhancements</h3>

    - Live CCTV Detection
    - GPS Tracking
    - Mobile Application
    - YOLO Object Detection
    - Smart Maintenance Dashboard

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    <h4>Developed using TensorFlow + Streamlit</h4>
    </div>
    """,
    unsafe_allow_html=True
)
