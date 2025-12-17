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
    <div style="
        background:linear-gradient(90deg,#1f2937,#111827);
        padding:2rem;
        border-radius:16px;
        text-align:center;
        margin-bottom:2rem;">
        <h1 style="margin-bottom:0.3rem;">{greeting} 👋</h1>
        <p style="opacity:0.85;">Welcome to the Facial Emotion Detection App</p>
        <br>
        <h1>I am a facial emotion detector</h1>
        <p>Let's see if you are looking happy today</p>
    </div>
    """,
    unsafe_allow_html=True
)

#sidebar
st.markdown("""
<style>
section[data-testid="stSidebar"] > div{
  background: linear-gradient(180deg,#0b1220,#111827);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] * { color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        "<div style='background:linear-gradient(90deg,#1f2937,#111827);"
        "padding:1rem;border-radius:12px;margin-bottom:1rem'>"
        "<b>⚙️ Settings</b><br><small>Customize the preview</small></div>",
        unsafe_allow_html=True
    )
    img_width = st.slider("Resize image", 150, 600, 320, 10)
left, right = st.columns([1.1, 1])

day_update = ["Good", "Average", "Bad", "Awsome Day"]
day_status = st.sidebar.multiselect("How is your day going", day_update)

# Upload card (visual box)
st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#1f2937,#111827);
        padding:1.5rem;
        border-radius:16px;
        text-align:center;
        margin-bottom:0.4rem;">
        <h3 style="margin:0">📸 Upload an image</h3>
        <p style="opacity:.85;margin:.4rem 0 0">
            Upload your most recent picture and I will tell if you are looking happy today
        </p>
        <small style="opacity:.7">JPG / JPEG / PNG • Clear face recommended</small>
    </div>
    """,
    unsafe_allow_html=True
)

# Real uploader (functionality)
uploaded = st.file_uploader(
    "",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=img_width)

    model = load_model()
    x = preprocess_image(img)

    label, confidence, probs = predict(model, x)
    #Output
  
    # --- Stylish result output ---
    lab = label.lower()
    
    if lab == "happy":
        emoji = "😢"
        title = "Sad"
        msg = "Oh no! You look sad. Everything okay?"
          
        if st.button("Tell me a joke 😄"):
            st.markdown(
                "**Why did the hairdresser win the race?**\n\n"
                "Because he knew a shortcut! ✂️😄"
            )
    elif lab == "sad":
        emoji = "🙂"
        title = "Happy"
        msg = "You are looking happy today. What's the secret?"
    else:
        emoji = ""
        title = label
        msg = label
      
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(90deg,#1f2937,#111827);
            padding:1.5rem;
            border-radius:16px;
            margin-top:1rem;">
            <h3 style="margin:0">{emoji} {title}</h3>
            <p style="margin:.4rem 0 0;opacity:.85">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    #Warning message
    st.markdown(
    "<div style='background:linear-gradient(90deg,#7c2d12,#451a03);"
    "padding:1rem;border-radius:12px;margin-top:1rem'>"
    "<b>⚠️ Important</b><br>"
    "<small>This is a personal machine learning project. If you receive an incorrect prediction, it may be due to dataset limitations, image quality, or the complexity of human emotions.</small>"
    "</div>",
    unsafe_allow_html=True
    )

st.markdown(
    "<div style='background:linear-gradient(90deg,#111827,#1f2937);"
    "padding:1rem;border-radius:12px;text-align:center;margin-top:2rem'>"
    "<small>Built by <b>Toahir Hussain</b></small><br>",
    unsafe_allow_html=True
)
