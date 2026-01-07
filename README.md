# Face Recognition POC (FastAPI + InsightFace)

A proof-of-concept microservice for face recognition using InsightFace (buffalo_l model). It provides endpoints to generate face embeddings from images and verify if two faces match.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

## Features

- Generate 512-dimensional face embeddings from uploaded images
- Verify face match using cosine similarity on stored embeddings
- Built with FastAPI for high performance
- Uses InsightFace buffalo_l model (ONNX-based, CPU-friendly)
- Model warmup on startup for fast inference
- CORS configured (currently allows all origins for testing – restrict in production)

## API Endpoints

### GET /

Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "model": "insightface-buffalo_l",
  "version": "1.0.0"
}
```

### POST /generate-embedding

Upload an image and generate a face embedding for the primary detected face.

**Request:**
- Multipart form with `image` file (JPEG/PNG)

**Response:**
```json
{
  "success": true,
  "embedding": [0.123, -0.456, ...],
  "model": "insightface-buffalo_l",
  "embedding_size": 512,
  "message": "Embedding generated successfully"
}
```

### POST /verify-face

Upload an image and compare it against a stored embedding to determine if faces match.

**Request:**
- Multipart form with image file and `stored_embedding` (list of 512 floats)

**Response:**
```json
{
  "success": true,
  "matched": true,
  "score": 0.92,
  "threshold": 0.85,
  "message": "Faces match"
}
```

## Local Development

### Clone the Repository

```bash
git clone https://github.com/JatinRajvani/face_poc_railway.git
cd face_poc_railway
```

### Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The first run will download InsightFace models (~326 MB) to `~/.insightface/models/`.

Access the interactive API documentation at `http://localhost:8000/docs` (Swagger UI).

