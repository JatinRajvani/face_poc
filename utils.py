import json
import numpy as np
import cv2
import logging
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "embeddings.json"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {'.jpg', '.jpeg', '.png'}

# -----------------------------
# Load InsightFace model ONCE
# -----------------------------
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("InsightFace model loaded successfully")


# -----------------------------
# Embedding storage helpers
# -----------------------------
def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_embeddings(data):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f)


# -----------------------------
# Image helpers
# -----------------------------
def validate_image_bytes(image_bytes):
    """Validate image bytes before processing"""
    if not image_bytes:
        raise ValueError("Image data is empty")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(f"Image size exceeds {MAX_IMAGE_SIZE} bytes")
    logger.info(f"Image validated - size: {len(image_bytes)} bytes")

def image_bytes_to_cv2(image_bytes):
    validate_image_bytes(image_bytes)
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or corrupted image file")
    logger.info(f"Image decoded - shape: {img.shape}")
    return img


# -----------------------------
# Face embedding generation
# -----------------------------
def generate_embedding(img):
    """
    Takes a BGR OpenCV image and returns embeddings for ALL detected faces
    Returns: list of 512-D face embeddings
    """
    faces = face_app.get(img)

    if not faces:
        logger.warning("No face detected in image")
        raise ValueError("No face detected in image")

    logger.info(f"Detected {len(faces)} face(s) in image")
    
    # Return embeddings for ALL detected faces
    embeddings = [face.embedding for face in faces]
    return embeddings

def generate_single_embedding(img):
    """
    Takes a BGR OpenCV image and returns only the first detected face embedding
    (For compatibility)
    Returns: single 512-D face embedding
    """
    embeddings = generate_embedding(img)
    logger.info("Returning first face embedding")
    return embeddings[0]


# -----------------------------
# Similarity calculation
# -----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

def warm_up_model():
    """Warm up the model with a dummy image for faster first inference"""
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        face_app.get(dummy_img)
        logger.info("✓ InsightFace model warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warm-up warning: {str(e)}")
