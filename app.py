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
st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="wide")

# ---------- Greeting ----------
hour = datetime.now().hour

if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

HEADER_CARD_HTML = f"""
<div style="background:linear-gradient(90deg,#1f2937,#111827);
padding:2rem;border-radius:16px;text-align:center;margin-bottom:1.5rem;">
  <h1 style="margin:0">{greeting} 👋</h1>
  <p style="opacity:.85;margin:.4rem 0 0">Welcome to the Facial Emotion Detection App</p>
</div>
"""

UPLOAD_CARD_HTML = """
<div style="background:linear-gradient(90deg,#0b1220,#111827);
padding:1.2rem;border-radius:12px;margin-bottom:1rem;">
  <h4 style="margin:0">📸 Upload an image</h4>
  <small style="display:block;margin-top:6px;opacity:.85;">
    Upload your most recent picture and I will predict Happy vs Sad.
  </small>
  <small style="display:block;margin-top:4px;opacity:.7;">
    JPG / JPEG / PNG • Clear face recommended
  </small>
</div>
"""

RESULT_CARD_HTML = """
<div style="background:linear-gradient(90deg,#0b1220,#111827);
padding:1.2rem;border-radius:12px;margin-bottom:1rem;">
  <h4 style="margin:0">📊 Result</h4>
  <small style="display:block;margin-top:6px;opacity:.75;">
    Prediction will appear here after upload.
  </small>
</div>
"""

left, right = st.columns([1.1, 1])

with left:
    st.markdown(UPLOAD_CARD_HTML, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=img_width)

with right:
    st.markdown(RESULT_CARD_HTML, unsafe_allow_html=True)

    if uploaded:
        model = load_model()
        x = preprocess_image(img)
        label, confidence, probs = predict(model, x)

        st.subheader(label)
        st.write(f"Confidence: **{confidence:.2%}**")
    else:
        st.info("Upload an image to see the prediction.")


st.markdown(
    "<div style='background:linear-gradient(90deg,#111827,#1f2937);"
    "padding:1rem;border-radius:12px;text-align:center;margin-top:2rem'>"
    "<small>Built by <b>Toahir Hussain</b></small><br>"
    "<a href='https://www.linkedin.com/in/toahirhussain/' target='_blank' "
    "style='color:#60a5fa;text-decoration:none;font-size:0.9rem'>"
    "🔗 LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True
)
