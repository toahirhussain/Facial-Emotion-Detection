import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pytz


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
st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="wide")

# ---------- Greeting ----------
# Streamlit Cloud often runs in UTC, so choose a timezone
# Detroit example: America/Detroit
tz = pytz.timezone("America/Detroit")
hour = datetime.now(tz).hour

if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

# ---------- Header ----------
st.title(f"{greeting}, welcome here 👋")
st.caption("Facial Emotion Detection • Happy 🙂 vs Sad 😢")
st.divider()

# ---------- Layout ----------
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Upload an image")
    st.write("Upload a clear face photo for the best result.")
    uploaded = st.file_uploader("Supported: JPG/JPEG/PNG", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=320)

with col_right:
    st.subheader("Prediction")
    if uploaded is None:
        st.info("Upload an image to see the prediction.")
    else:
        model = load_model()
        x = preprocess_image(img)

        label, confidence, probs = predict(model, x)

        # Your message logic (you had it reversed)
        if label.lower() == "happy":
            emotion_msg = "You are looking happy today. What's the secret? 😊"
        elif label.lower() == "sad":
            emotion_msg = "Oh no! You are looking sad. Everything okay? 💛"
        else:
            emotion_msg = label  # fallback safety

        st.success(emotion_msg)
        st.metric("Confidence", f"{confidence:.2%}")

        with st.expander("Class probabilities", expanded=True):
            for name, p in zip(CLASS_NAMES, probs):
                st.write(f"**{name}**: {p:.2%}")

        st.warning(
            "⚠️ This is a personal machine learning project. Results may be inaccurate due to dataset limitations, "
            "image quality, and the complexity of human emotions."
        )

st.divider()
st.caption("Built by Toahir Hussain • Facial Emotion Detection")
