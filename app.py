import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime


# ---------------------------
# CONFIG (edit if needed)
# ---------------------------
IMG_SIZE = (256, 256)                 # MUST match training image size
CLASS_NAMES = ["Sad", "Happy"]        # MUST match training label order
MODEL_PATH = "happysadmodelFinal.h5"
  # put this file in the same folder as app.py
# If your model is elsewhere, use full path like:
# MODEL_PATH = r"C:\Users\Toahir Tayef\emotion-app\happysadmodelFinal.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.info("Tip: Put happysadmodelFinal.h5 in the same folder as app.py, or set MODEL_PATH to the full path.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # must match training normalization
    arr = np.expand_dims(arr, axis=0)              # (1, H, W, 3)
    return arr

def predict(model, x):
    preds = model.predict(x, verbose=0)

    # Handles both sigmoid (binary output) and softmax (2-class output)
    if preds.shape[-1] == 1:
        # Assumption: model outputs probability of class 1 (second class in CLASS_NAMES)
        prob_class1 = float(preds[0][0])
        probs = [1.0 - prob_class1, prob_class1]
    else:
        probs = preds[0].tolist()

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ---------- Page config MUST be first ----------
st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="centered")

# ---------- Greeting ----------
hour = datetime.now().hour

if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

st.markdown(
    f"""
    <style>
    .header {{
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
    }}
    </style>

    <div class="header">
        <h1>{greeting} 👋</h1>
        <p>Welcome to the Facial Emotion Detection App</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.markdown(
    f"""
    <style>
    .header {{
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
    }}
    </style>

    <div class="header">
        <h1>I am a facial emotion detector</h1>
        <p>Let's see if you are looking happy toda</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.text("Upload your most recent picture and I will tell if you are looking happy today.")

uploaded = st.file_uploader("Please upload an image (jpg)", type=["jpg"])
img_width = st.sidebar.slider("Resize your mage", 150, 600, 320, 10)
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=img_width)

    model = load_model()
    x = preprocess_image(img)

    label, confidence, probs = predict(model, x)

    if label.lower() == "happy":
        emotion = "Oh no! You are looking so sad! Everything okay?"
    elif label.lower() == "sad":
        emotion = "You are looking happy today. What's the secrect?"
    else:
        emotion = label  # fallback safety
    st.subheader(f"**{emotion}**")
    st.divider()
    st.write("⚠️ This is a personal machine learning project. Predictions may be inaccurate due to dataset limitations, image quality, and the inherent complexity of human emotions.")
st.divider()
st.caption("Built by Toahir Hussain • Facial Emotion Detection")
