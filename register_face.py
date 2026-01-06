# # register_face.py
# from utils import load_embeddings, save_embeddings, image_bytes_to_cv2, generate_embedding

# def register_face(student_id: str, image_bytes: bytes):
#     img = image_bytes_to_cv2(image_bytes)

#     # generate embedding using your existing InsightFace code
#     embedding = generate_embedding(img)   # already in your code

#     data = load_embeddings()
#     data[student_id] = embedding.tolist()
#     save_embeddings(data)

#     return {
#         "message": "Face registered successfully",
#         "studentId": student_id
#     }


import logging
from utils import image_bytes_to_cv2, generate_single_embedding

logger = logging.getLogger(__name__)

def register_face(image_bytes):
    """
    Register a face by extracting embedding from image.
    Returns embedding for the primary/first detected face.
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        dict with embedding, model info, and embedding size
    """
    try:
        logger.info("Starting face registration")
        img = image_bytes_to_cv2(image_bytes)
        embedding = generate_single_embedding(img)  # First face only for registration
        
        result = {
            "success": True,
            "embedding": embedding.tolist(),
            "model": "insightface-buffalo_l",
            "embedding_size": len(embedding),
            "message": "Face registered successfully"
        }
        logger.info("Face registration completed successfully")
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}", exc_info=True)
        raise
