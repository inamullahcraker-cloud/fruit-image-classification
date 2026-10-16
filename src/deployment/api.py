from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predict import Predictor
from src.utils.device import get_device
from src.config.config import load_config

app = FastAPI(
    title="Fruit Image Classification API",
    version="1.0.0",
)

config=load_config("configs/config.yaml")
device = get_device(config.training.device,)
predictions= Predictor(
        checkpoint_path=config.inference.checkpoint_path,
        class_mapping_path=config.inference.class_mapping,
        model_name=config.model.name,
        image_size=config.dataset.image_size,
        device=device,
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

    result = predictions(
        image_path=image_path)

    return result
