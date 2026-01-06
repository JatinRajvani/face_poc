import logging
from fastapi import FastAPI, UploadFile, File, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from register_face import register_face
from verify_face import verify_face
from utils import warm_up_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================
# Pydantic Response Models
# =====================
class EmbeddingResponse(BaseModel):
    """Response for embedding generation"""
    success: bool
    embedding: List[float]
    model: str
    embedding_size: int
    message: str

class VerifyResponse(BaseModel):
    """Response for face verification"""
    success: bool
    matched: bool
    score: float
    threshold: float
    message: str

class HealthResponse(BaseModel):
    """Response for health check"""
    status: str
    model: str
    version: str

# =====================
# FastAPI App Setup
# =====================
app = FastAPI(
    title="Face Recognition Service",
    description="Microservice for face embedding generation and verification",
    version="1.0.0"
)

# =====================
# CORS Configuration
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")

# =====================
# Startup/Shutdown Events
# =====================
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("========== Application Starting ==========")
    try:
        warm_up_model()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("========== Application Shutting Down ==========")

# =====================
# API Endpoints
# =====================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {
        "status": "running",
        "model": "insightface-buffalo_l",
        "version": "1.0.0"
    }

@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding_api(image: UploadFile = File(...)):
    """
    Generate face embedding from image.
    Returns embedding for the primary detected face.
    
    - **image**: Image file (JPEG/PNG)
    
    Returns:
        - **success**: Operation status
        - **embedding**: 512-dimensional face embedding
        - **model**: Model name used
        - **embedding_size**: Size of embedding vector
    """
    try:
        logger.info(f"Embedding generation requested for file: {image.filename}")
        
        # Validate file type
        if not image.content_type.startswith('image/'):
            logger.warning(f"Invalid content type: {image.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG/PNG)"
            )
        
        image_bytes = await image.read()
        result = register_face(image_bytes)
        logger.info("Embedding generation completed successfully")
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during embedding generation"
        )

@app.post("/verify-face", response_model=VerifyResponse)
async def verify_face_api(
    image: UploadFile = File(...),
    stored_embedding: List[float] = Body(...)
):
    """
    Verify if a photo matches a stored face embedding.
    
    - **image**: Image file (JPEG/PNG)
    - **stored_embedding**: 512-dimensional stored face embedding (list of 512 floats)
    
    Returns:
        - **success**: Operation status
        - **matched**: Whether faces match (True/False)
        - **score**: Cosine similarity score (0-1)
        - **threshold**: Matching threshold used
        - **message**: Result message
    """
    try:
        logger.info(f"Face verification requested for file: {image.filename}")
        
        # Validate file type
        if not image.content_type.startswith('image/'):
            logger.warning(f"Invalid content type: {image.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG/PNG)"
            )
        
        image_bytes = await image.read()
        result = verify_face(image_bytes, stored_embedding)
        logger.info("Face verification completed successfully")
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during verification"
        )

logger.info("FastAPI application initialized successfully")







