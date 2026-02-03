import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import streamlit.components.v1 as components

# -----------------------------
# CONFIG (MATCHES NOTEBOOK)
# -----------------------------
IMAGE_SIZE = 256
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Potato Leaf Disease Detection System",
    page_icon="🌱",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_disease_model.keras",compile=False)

model = load_model()

# ================= BACKGROUND =================
def set_background(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

# ================= CSS =================
st.markdown("""
<style>
.green-card {
    background: linear-gradient(
        135deg,
        rgba(220,252,231,0.92),
        rgba(187,247,208,0.88)
    );
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.25);
    margin-bottom: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #064e3b;
    margin-bottom: 8px;
}

.text {
    font-size: 14.5px;
    color: #065f46;
    line-height: 1.5;
}

.glass-strip {
    background: linear-gradient(
        135deg,
        rgba(22,101,52,0.65),
        rgba(6,95,70,0.55)
    );
    padding: 14px 18px;
    border-radius: 14px;
    color: #ecfdf5;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
}

.footer {
    font-size: 12px;
    color: #064e3b;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.action-box {
    background: linear-gradient(
        135deg,
        rgba(163, 230, 53, 0.35),
        rgba(132, 204, 22, 0.30)
    );
    padding: 14px 16px;
    border-radius: 14px;
    margin-top: 12px;
    color: #ecfccb;
}

.action-box ul {
    margin: 8px 0 0 18px;
}

.action-box li {
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style="text-align:center; margin-bottom:25px;">
    <h1 style="font-size:42px; font-weight:800; color:#ecfdf5;">
        🌱 Potato Leaf Disease Detection System
    </h1>
    <p style="font-size:16px; color:#d1fae5;">
        AI-powered system for detecting potato leaf diseases using Deep Learning (CNN)
    </p>
</div>
""", unsafe_allow_html=True)

# ================= MAIN LAYOUT =================
left, right = st.columns([1.15, 1])

# ================= LEFT PANEL =================
with left:
    st.markdown("""
    <div class="green-card">
        <div class="section-title">📘 About the System</div>
        <p class="text">
        This system uses a <b>Convolution Neural Network (CNN)</b> trained on the
        <b>PlantVillage dataset</b> to automatically identify diseases in potato leaves.
        The goal is to assist farmers, students, and researchers in detecting diseases
        early and taking appropriate action.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="green-card">
        <div class="section-title">🦠 Supported Classes</div>
        <ul class="text">
            <li><b>Early Blight</b> – causes brown spots and leaf damage</li>
            <li><b>Late Blight</b> – highly destructive disease requiring immediate action</li>
            <li><b>Healthy Leaf</b> – no visible disease symptoms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="green-card">
        <div class="section-title">📋 How to Use This System</div>
        <ol class="text">
            <li>Capture a <b>clear image</b> of a potato leaf</li>
            <li>Ensure <b>good lighting</b> and minimal background</li>
            <li>Upload the image below</li>
            <li>View prediction result and confidence score</li>
            <li>Follow the recommended action provided</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ================= RIGHT PANEL =================
with right:
    st.markdown("""
    <div class="green-card">
        <div class="section-title">📤 Upload Image</div>
        <p class="text">JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        img_col, result_col = st.columns([1, 1.4])

        # IMAGE PREVIEW
        with img_col:
            st.markdown("""
            <div class="green-card">
                <div class="section-title">📷 Uploaded Leaf Image</div>
            """, unsafe_allow_html=True)

            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # MODEL PREDICTION (UNCHANGED LOGIC)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = round(100 * np.max(predictions[0]), 2)
        predicted_class = CLASS_NAMES[predicted_index]

        if predicted_class == "Potato___Early_blight":
            disease = "Early Blight"
            actions = [
                "Remove affected leaves",
                "Apply recommended fungicide",
                "Avoid overhead irrigation"
            ]
        elif predicted_class == "Potato___Late_blight":
            disease = "Late Blight"
            actions = [
                "Remove infected plants",
                "Apply fungicide immediately",
                "Improve airflow and drainage"
            ]
        else:
            disease = "Healthy Leaf"
            actions = [
                "No treatment required",
                "Monitor plant health regularly",
                "Maintain proper irrigation"
            ]

        # RESULTS


        with result_col:
            actions_html = "".join(f"<li>{a}</li>" for a in actions)

            components.html(
                f"""
                <style>
                    .green-card {{
                        background: linear-gradient(
                            135deg,
                            rgba(220,252,231,0.92),
                            rgba(187,247,208,0.88)
                        );
                        padding: 18px;
                        border-radius: 16px;
                        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
                        font-family: sans-serif;
                    }}

                    .section-title {{
                        font-size: 19px;
                        font-weight: 700;
                        color: #064e3b;
                        margin-bottom: 12px;
                    }}

                    .glass-strip {{
                        background: linear-gradient(
                            135deg,
                            rgba(22,101,52,0.65),
                            rgba(6,95,70,0.55)
                        );
                        padding: 12px 16px;
                        border-radius: 12px;
                        color: #ecfdf5;
                        font-size: 15px;
                        font-weight: 600;
                        margin-bottom: 10px;
                    }}

                    .action-box {{
                        background: rgba(16,185,129,0.25);
                        padding: 12px 14px;
                        border-radius: 12px;
                        color: #064e3b;
                    }}

                    ul {{
                        margin: 6px 0 0 18px;
                    }}
                </style>

                <div class="green-card">
                    <div class="section-title">💡 Prediction & Action</div>

                    <div class="glass-strip">
                        🦠 Detected Disease: <b>{disease}</b>
                    </div>

                    <div class="glass-strip">
                        📊 Confidence Score: <b>{confidence}%</b>
                    </div>

                    <div class="action-box">
                        <b>Recommended Action</b>
                        <ul>
                            {actions_html}
                        </ul>
                    </div>
                </div>
                """,
                height=320
            )


            



# ================= FOOTER =================
st.markdown("""
<div class="green-card footer">
<b>Disclaimer:</b> For educational and decision-support purposes only.
</div>
""", unsafe_allow_html=True)

