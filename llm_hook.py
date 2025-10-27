# llm_hooks.py
import os
from typing import List, Tuple

# Optional Groq integration via langchain_groq
try:
    from langchain_groq import ChatGroq  # may not exist in all environments
    from langchain_core.messages import HumanMessage
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

def analyze_for_user(detections: List[Tuple[int,int,int,int,float]], filename: str = None) -> str:
    """
    Use an LLM (Groq via ChatGroq) to:
      - state the number of faces (the LLM's wording),
      - provide a short description of the photo and suggestions.

    We pass structured metadata (count, bounding boxes). If GROQ_API_KEY not present
    or ChatGroq unavailable, return a deterministic fallback summary.
    """
    count = len(detections)
    boxes_s = []
    for (x, y, w, h, conf) in detections:
        boxes_s.append(f"({x},{y},{w},{h})")
    boxes_text = ", ".join(boxes_s) if boxes_s else "none"

    groq_api_key = os.getenv("GROQ_API_KEY")

    prompt = f"""
You are given detection metadata extracted from an image{(f' named {filename}' if filename else '')}.
Detected face count: {count}
Bounding boxes (x,y,w,h): {boxes_text}

Please:
1) State the detected face count in a single short sentence.
2) Provide a 1-2 sentence human-friendly description of the image scene (based on the detection metadata and typical photo patterns).
3) Give one suggestion to improve detection quality (lighting/angle) in one short sentence.

Keep the total response under 120 tokens. Be concise and friendly.
"""

    # If Groq is configured and available, call it
    if groq_api_key and HAS_GROQ:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=150
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            # degrade gracefully
            return fallback_summary(count, boxes_text, error=str(e))
    else:
        return fallback_summary(count, boxes_text)


def fallback_summary(count: int, boxes_text: str, error: str = None) -> str:
    """
    Deterministic fallback text when LLM isn't available.
    """
    lines = []
    lines.append(f"Detected {count} face{'s' if count != 1 else ''}.")
    if count > 0:
        lines.append(f"Bounding boxes: {boxes_text}.")
    else:
        lines.append("No faces were detected in this photo.")
    lines.append("Description: A typical photo with subjects facing the camera.")
    lines.append("Suggestion: Improve lighting and try a frontal angle for better detection.")
    if error:
        lines.append(f"(LLM unavailable: {error})")
    return " ".join(lines)
