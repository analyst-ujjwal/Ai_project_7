# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from detector import detect_faces
from llm_hook import analyze_for_user
import io
import os

st.set_page_config(page_title="Face Count & Describe", layout="centered")
st.title("7️⃣ Face Detection App (OpenCV + Groq Inference)")
st.markdown(
    "Real-time detection with Groq acceleration. "
    "Upload an image; OpenCV detects faces, and an LLM provides face count "
    "and a short description of the photo."
)


uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Read as PIL Image (safe)
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)

    # Detect faces
    try:
        detections = detect_faces(img)
    except Exception as e:
        st.error(f"Detection failed: {e}")
        st.stop()

    # Show boxes on a copy for visual feedback
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    display = img_bgr.copy()
    for (x, y, w, h, conf) in detections:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, f"{conf:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    st.image(display_rgb, caption=f"Detected {len(detections)} face(s)", use_column_width=True)

    # Ask LLM to produce count & description
    filename = getattr(uploaded, "name", None)
    try:
        analysis = analyze_for_user(detections, filename=filename)
    except Exception as e:
        analysis = f"Analysis failed: {e}"

    st.subheader("LLM Report")
    st.write(analysis)

else:
    st.info("Upload an image to start. (Tip: frontal group photos work best for Haar cascade.)")
