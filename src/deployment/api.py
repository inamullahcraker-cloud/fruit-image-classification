from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predict import predict_image
from src.utils.device import get_device

app = FastAPI(
    title="Fruit Image Classification API",
    version="1.0.0",
)

MODEL_NAME = "resnet18"
IMAGE_SIZE = 224

CHECKPOINT = Path(
    "artifacts/checkpoints/best_model.pth"
)

UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASS_NAMES = []

DEVICE = get_device("auto")


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    image_path = UPLOAD_DIR / file.filename

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    result = predict_image(
        image_path=image_path,
        checkpoint_path=CHECKPOINT,
        class_names=CLASS_NAMES,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        device=DEVICE,
    )

    return result
