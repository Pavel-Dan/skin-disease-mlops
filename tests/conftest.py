import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
os.environ.setdefault("SKIP_MINIO", "true")
os.environ.setdefault("MODEL_ARCH", "resnet18")

from pathlib import Path

import pytest

from api.main import load_model
from scripts.generate_demo_data import generate_processed_manifest
from src.training.train import train_model


@pytest.fixture(scope="session", autouse=True)
def bootstrap_model() -> None:
    generate_processed_manifest(n_per_split=2)
    checkpoint = Path("models/best_model.pt")
    if checkpoint.exists():
        checkpoint.unlink()
    train_model(epochs=1, batch_size=4, register_model=False)
    load_model()
