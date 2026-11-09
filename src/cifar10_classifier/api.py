"""API local de inferência da CNN CIFAR-10."""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from .serialization import DEFAULT_MODEL_DIR, MODEL_FILE, verify_export

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


def _decode_image(payload: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in IMAGE_FORMATS or getattr(image, "is_animated", False):
                raise HTTPException(415, "Envie uma imagem JPEG, PNG ou WebP estática")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "A imagem excede o limite de 16 megapixels")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            resized = ImageOps.fit(rgb, (32, 32), method=Image.Resampling.BILINEAR)
            return np.asarray(resized)
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "A imagem excede o limite de 16 megapixels") from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(415, "Envie uma imagem JPEG, PNG ou WebP válida") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(400, "Não foi possível decodificar a imagem") from exc


def _load_predictor(directory: Path):
    manifest = verify_export(directory)

    from tensorflow import keras

    model = keras.models.load_model(directory / MODEL_FILE, compile=False)
    if model.input_shape != (None, 32, 32, 3) or model.output_shape != (None, len(manifest["classes"])):
        raise ValueError("A assinatura do modelo exportado é incompatível com a API")
    return model, manifest


def create_app(model_dir: Path = DEFAULT_MODEL_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model, application.state.manifest = _load_predictor(model_dir)
        yield

    application = FastAPI(title="CIFAR-10 — Classificação de imagens", lifespan=lifespan)

    @application.get("/health")
    def health(request: Request) -> dict:
        manifest = request.app.state.manifest
        return {
            "status": "ok",
            "model": manifest["run_name"],
            "checkpoint_epoch": manifest["best_epoch"],
        }

    @application.post("/predict")
    def predict(request: Request, file: Annotated[UploadFile, File()]) -> Prediction:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "O arquivo excede o limite de 5 MiB")
        image = _decode_image(payload)
        scores = np.asarray(request.app.state.model(image[None, ...], training=False))[0]
        classes = request.app.state.manifest["classes"]
        class_id = int(np.argmax(scores))
        return Prediction(
            class_id=class_id,
            class_name=classes[class_id],
            confidence=float(scores[class_id]),
            probabilities={name: float(score) for name, score in zip(classes, scores, strict=True)},
        )

    return application


app = create_app()
