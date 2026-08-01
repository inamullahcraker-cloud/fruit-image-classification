from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.inference.predict import predict_image


def batch_predict(
    image_dir: str | Path,
    checkpoint_path: str | Path,
    class_names: list[str],
    model_name: str,
    image_size: int,
    device,
    output_csv: str | Path = "reports/predictions.csv",
) -> pd.DataFrame:

    image_dir = Path(image_dir)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            }
        ]
    )

    results = []

    for image_path in image_paths:

        start = time.perf_counter()

        prediction = predict_image(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            class_names=class_names,
            model_name=model_name,
            image_size=image_size,
            device=device,
        )

        inference_time = time.perf_counter() - start

        results.append(
            {
                "filename": image_path.name,
                "predicted_class": prediction["class"],
                "class_index": prediction["class_index"],
                "confidence": prediction["confidence"],
                "inference_time_ms": inference_time * 1000,
            }
        )

    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    df.to_csv(
        output_csv,
        index=False,
    )

    return df
