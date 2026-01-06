# # verify_face.py
# from utils import load_embeddings, image_bytes_to_cv2, cosine_similarity,generate_embedding

# THRESHOLD = 0.45

# def verify_face(student_id: str, image_bytes: bytes):
#     data = load_embeddings()

#     if student_id not in data:
#         return {
#             "matched": False,
#             "reason": "Face not registered"
#         }

#     img = image_bytes_to_cv2(image_bytes)
#     current_embedding = generate_embedding(img)

#     stored_embedding = data[student_id]
#     score = cosine_similarity(current_embedding, stored_embedding)

#     return {
#         "matched": score >= THRESHOLD,
#         "score": float(score)
#     }


import logging
import numpy as np
from utils import image_bytes_to_cv2, generate_single_embedding, cosine_similarity

logger = logging.getLogger(__name__)

THRESHOLD = 0.52

def validate_embedding(embedding):
    """
    Validate embedding format
    
    Args:
        embedding: list or array of floats
        
    Raises:
        ValueError: if embedding is invalid
    """
    if not embedding:
        raise ValueError("Embedding is empty")
    if not isinstance(embedding, (list, np.ndarray)):
        raise ValueError("Embedding must be a list or array")
    if len(embedding) != 512:
        raise ValueError(f"Embedding size must be 512, got {len(embedding)}")
    logger.info("Embedding validation passed")

def verify_face(image_bytes, stored_embedding):
    """
    Verify if a photo matches a stored embedding.
    
    Args:
        image_bytes: Binary image data
        stored_embedding: List of 512 floats (stored face embedding)
        
    Returns:
        dict with match result and similarity score
    """
    try:
        logger.info("Starting face verification")
        
        # Validate embedding
        validate_embedding(stored_embedding)
        
        # Extract embedding from image
        img = image_bytes_to_cv2(image_bytes)
        current_embedding = generate_single_embedding(img)  # First face only
        
        # Calculate similarity
        score = cosine_similarity(current_embedding, stored_embedding)
        matched = score >= THRESHOLD
        
        logger.info(f"Face verification completed - matched: {matched}, score: {score:.4f}")
        
        return {
            "success": True,
            "matched": matched,
            "score": float(score),
            "threshold": THRESHOLD,
            "message": "Match found" if matched else "No match"
        }
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during verification: {str(e)}", exc_info=True)
        raise


