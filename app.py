import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Emotion Classifier", page_icon="😊")
st.title("Hello, welcome here")
st.title("Emotion Classifier (Happy 😊 vs Sad) 😭")
st.write("Upload an image and the model will predict whether the person looks **Happy** or **Sad**.")

uploaded = st.file_uploader("Please upload an image (jpg)", type=["jpg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    x = preprocess_image(img)

    label, confidence, probs = predict(model, x)

    st.subheader(f"This person looks: **{label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.write("Class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"- {name}: {p:.2%}")

st.markdown('<div class="footer">Built by: <b>Toahir Hussain</b>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload an image to predict emotion.")