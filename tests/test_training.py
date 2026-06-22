from pathlib import Path

import torch

from scripts.generate_demo_data import generate_processed_manifest
from src.training.dataset import create_dataloaders
from src.training.model import build_model
from src.training.train import set_seed, train_model
from src.utils.config import CLASSES, MODEL_PATH, SEED


def test_model_forward_one_batch(tmp_path: Path) -> None:
    generate_processed_manifest(n_per_split=2)
    loaders = create_dataloaders(batch_size=2)
    model = build_model()
    images, labels = next(iter(loaders["train"]))
    outputs = model(images)
    assert outputs.shape[0] == labels.shape[0]
    assert outputs.shape[1] == len(CLASSES)


def test_training_smoke(tmp_path: Path, monkeypatch) -> None:
    csv_path = generate_processed_manifest(n_per_split=2)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    set_seed(SEED)
    history = train_model(csv_path=csv_path, epochs=1, batch_size=2, register_model=False)
    assert "best_val_f1_macro" in history
    assert MODEL_PATH.exists()


def test_seed_reproducibility() -> None:
    set_seed(SEED)
    a = torch.randint(0, 100, (3,))
    set_seed(SEED)
    b = torch.randint(0, 100, (3,))
    assert torch.equal(a, b)
