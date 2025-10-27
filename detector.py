# detector.py
import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Union

# Path to Haar cascade
BASE = os.path.dirname(__file__)
CASCADE_PATH = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"Haarcascade not found at {CASCADE_PATH}. "
        "Place 'haarcascade_frontalface_default.xml' inside the models/ folder."
    )

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

ImageLike = Union[np.ndarray, Image.Image]

def to_bgr_array(img: ImageLike) -> np.ndarray:
    """
    Convert a PIL.Image or numpy array to an OpenCV BGR uint8 image.
    Handles grayscale, RGB, RGBA, and already-BGR images.
    """
    if isinstance(img, np.ndarray):
        arr = img.astype(np.uint8) if img.dtype != np.uint8 else img
        if arr.ndim == 2:  # Grayscale
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 3:
            # Heuristic: if mean R < mean B, assume RGB → BGR
            if np.mean(arr[:, :, 0]) < np.mean(arr[:, :, 2]):
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        elif arr.shape[2] == 4:  # RGBA → BGR
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError("Unsupported numpy image shape.")
    elif isinstance(img, Image.Image):
        arr = np.array(img).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("Image must be a numpy array or PIL.Image object.")

def detect_faces(
    img: ImageLike,
    scaleFactor: float = 1.1,
    minNeighbors: int = 5,
    minSize: Tuple[int, int] = (30, 30)
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces in an image using Haar cascade.
    Returns a list of tuples (x, y, w, h, confidence).
    Haar cascades don't return confidence, so we set confidence=1.0.
    """
    img_bgr = to_bgr_array(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in rects]
