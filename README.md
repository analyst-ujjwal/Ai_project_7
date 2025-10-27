# 7️⃣ Face Detection App (OpenCV + Groq Inference)

Real-time face detection with Groq acceleration. Upload an image, OpenCV detects faces, and an LLM provides the face count along with a short, human-friendly description of the photo.

---

## Features

- **Face Detection:** Uses OpenCV Haar Cascade to detect faces in any uploaded image.
- **LLM Analysis:** The detected face data is fed to a Groq-powered LLaMA model (or local fallback) to generate:
  - Face count in a natural sentence.
  - Short description of the photo scene.
  - Suggestion for improving detection quality.
- **Real-Time Visualization:** Bounding boxes are drawn on detected faces for visual feedback.
- **Easy to Use:** Upload an image via Streamlit interface and see instant results.

---

## Folder Structure

project-7/
├── app.py # Streamlit frontend
├── detector.py # OpenCV face detection
├── llm_hooks.py # LLM integration for face count & description
├── models/
│ └── haarcascade_frontalface_default.xml # Haar cascade file
├── requirements.txt
├── README.md
└── .env # Optional: GROQ_API_KEY for cloud inference

yaml


---

## Setup Instructions

1. **Clone the repository**:

```bash
git clone
cd project-7
Create virtual environment:

bash
python -m venv venv

source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Install dependencies:

bash


pip install -r requirements.txt
Download Haar Cascade (if not included):

Download haarcascade_frontalface_default.xml

Place it inside the models/ folder.

Add Groq API Key for LLM inference:

init

GROQ_API_KEY=your_groq_api_key_here
Run the app:

bash

streamlit run app.py
Open the URL displayed in your terminal (usually http://localhost:8501).

