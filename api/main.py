from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.training.model import build_model, get_model_transforms
from src.utils.config import (
    CLASSES,
    INTERIM_DIR,
    LOG_PREDICTIONS_PATH,
    MEDICAL_DISCLAIMER,
    MODEL_PATH,
    MODEL_NAME,
)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/jpg"}

_model: torch.nn.Module | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_model_transforms(train=False)
_model_info: dict[str, Any] = {}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    confidence_percent: float
    top3_predictions: list[dict[str, float | str]]
    disclaimer: str


def _load_checkpoint() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    return torch.load(MODEL_PATH, map_location=_device, weights_only=False)


def load_model() -> None:
    global _model, _model_info
    checkpoint = _load_checkpoint()
    arch = checkpoint.get("model_arch")
    model = build_model(pretrained=False, arch=arch).to(_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model = model
    _model_info = {
        "model_name": MODEL_NAME,
        "checkpoint_path": str(MODEL_PATH),
        "classes": checkpoint.get("classes", CLASSES),
        "val_f1_macro": checkpoint.get("val_f1_macro"),
    }
    metrics_path = INTERIM_DIR / "test_metrics.json"
    if metrics_path.exists():
        import json

        _model_info["test_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        _model_info["status"] = "model_not_loaded"
    yield


app = FastAPI(
    title="Skin Lesion Classifier API",
    description="Bloc 4 MLOps API for multi-class skin lesion classification.",
    version="0.1.0",
    lifespan=lifespan,
)


def _validate_upload(file: UploadFile, content: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or BMP.")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5 MB.")


def _predict_tensor(image: Image.Image) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    ranked = sorted(
        [{"class": CLASSES[i], "probability": float(probabilities[i])} for i in range(len(CLASSES))],
        key=lambda item: item["probability"],
        reverse=True,
    )
    best = ranked[0]
    response = PredictionResponse(
        predicted_class=str(best["class"]),
        confidence=float(best["probability"]),
        confidence_percent=round(float(best["probability"]) * 100, 2),
        top3_predictions=ranked[:3],
        disclaimer=MEDICAL_DISCLAIMER,
    )
    _log_prediction(response)
    return response


def _log_prediction(response: PredictionResponse) -> None:
    LOG_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PREDICTIONS_PATH.exists()
    with LOG_PREDICTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "predicted_class", "confidence"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predicted_class": response.predicted_class,
                "confidence": response.confidence,
            }
        )


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok" if _model is not None else "degraded"
    return {"status": status, "model_loaded": str(_model is not None)}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not _model_info:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable.")
    return _model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    _validate_upload(file, content)
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    return _predict_tensor(image)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})
