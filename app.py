from __future__ import annotations

from pathlib import Path

import uvicorn

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import load_config
from src.inference.predict import Predictor
from src.utils.device import get_device

app = FastAPI(
    title="Fruit Image Classification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
config = load_config("configs/config.yaml")

device = get_device(config.training.device)

predictor = Predictor(
    checkpoint_path=config.inference.checkpoint_path,
    class_mapping_path=config.inference.class_mapping,
    model_name=config.model.name,
    image_size=config.dataset.image_size,
    device=device,
    confidence_threshold=config.inference.confidence_threshold,
)
config = load_config("configs/config.yaml")

device = get_device(config.training.device)

checkpoint_path = (
    Path(config.experiment.output_dir)
    / "checkpoints"
    / "best_model.pth"
)


@app.get("/")
def home():

    return {
        "message": "Fruit Classification API",
        "status": "running",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    upload_dir = Path("uploads")

    upload_dir.mkdir(
        exist_ok=True
    )

    image_path = upload_dir / file.filename

    with image_path.open(
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    prediction = predictor.predict(
    image_path=image_path,)


    return prediction


if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )