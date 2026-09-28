from fastapi.responses import FileResponse
import os
import tempfile

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_video


app = FastAPI(
    title="Human Activity Detection API",
    description="CNN-LSTM based Human Activity Recognition",
    version="1.0.0"
)

# Allow browser frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CLASSES = [
    "Boxing",
    "Handclapping",
    "Handwaving",
    "Jogging",
    "Running"
]


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {".avi", ".mp4", ".mov", ".mkv"}

    filename = file.filename or "uploaded_video"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use AVI, MP4, MOV or MKV."
        )

    temp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = predict_video(temp_path)

        return {
            "filename": filename,
            "activity": result["activity"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "device": result.get("device", "cpu")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)