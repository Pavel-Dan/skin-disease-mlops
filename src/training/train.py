from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import create_dataloaders
from src.training.model import build_model
from src.data.data_sync import ensure_training_data
from src.utils.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BATCH_SIZE,
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IDX_TO_CLASS,
    INTERIM_DIR,
    LEARNING_RATE,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_ARCH,
    MODEL_PATH,
    MODELS_DIR,
    SEED,
)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_mlflow() -> None:
    import os

    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _load_class_weights() -> torch.Tensor | None:
    weights_path = INTERIM_DIR / "class_weights_train.json"
    if not weights_path.exists():
        return None
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    return torch.tensor([weights[label] for label in CLASSES], dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
        images = images.to(device)
        labels = labels.to(device)
        if train and optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        preds.extend(outputs.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return float(np.mean(losses)), float(f1)


def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auc_ovr"] = 0.0
    return metrics


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[list[float]] = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probabilities.argmax(axis=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        probs.extend(probabilities.tolist())
    return y_true, y_pred, np.array(probs)


def train_model(
    csv_path: Path | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    experiment_name: str = "skin-classifier",
    register_model: bool = True,
) -> dict:
    set_seed()
    _configure_mlflow()
    ensure_training_data(from_minio=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders(csv_path=csv_path, batch_size=batch_size)
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("Train and validation splits are required for training.")

    model = build_model().to(device)
    class_weights = _load_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1.0
    best_state: dict | None = None
    patience_counter = 0
    history: dict = {"epochs": []}

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="train"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": SEED,
                "device": str(device),
                "model_arch": MODEL_ARCH,
            }
        )
        for epoch in range(1, epochs + 1):
            train_loss, train_f1 = _run_epoch(
                model, loaders["train"], criterion, optimizer, device, train=True
            )
            val_loss, val_f1 = _run_epoch(
                model, loaders["val"], criterion, None, device, train=False
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_f1_macro": train_f1,
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1,
                },
                step=epoch,
            )
            history["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_f1_macro": train_f1,
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1,
                }
            )
            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": best_state,
            "classes": CLASSES,
            "idx_to_class": IDX_TO_CLASS,
            "val_f1_macro": best_f1,
            "model_arch": MODEL_ARCH,
        }
        torch.save(checkpoint, MODEL_PATH)
        mlflow.log_metric("best_val_f1_macro", best_f1)
        log_kwargs = {
            "pytorch_model": model,
            "artifact_path": "model",
            "serialization_format": "pickle",
        }
        if register_model:
            log_kwargs["registered_model_name"] = MODEL_NAME
        mlflow.pytorch.log_model(**log_kwargs)

        if "test" in loaders:
            y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)
            test_metrics = compute_metrics(y_true, y_pred, y_prob)
            mlflow.log_metrics(
                {
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1_macro": test_metrics["f1_macro"],
                    "test_precision_macro": test_metrics["precision_macro"],
                    "test_recall_macro": test_metrics["recall_macro"],
                    "test_auc_ovr": test_metrics.get("auc_ovr", 0.0),
                }
            )
            metrics_path = INTERIM_DIR / "test_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            history["test_metrics"] = test_metrics

        if register_model:
            try:
                client = mlflow.tracking.MlflowClient()
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                if versions:
                    latest = max(versions, key=lambda v: int(v.version))
                    if best_f1 >= float(latest.tags.get("val_f1_macro", 0) or 0):
                        client.transition_model_version_stage(
                            MODEL_NAME, latest.version, "Production", archive_existing_versions=True
                        )
            except Exception as exc:
                print(f"Model registry promotion skipped: {exc}")

    history["best_val_f1_macro"] = best_f1
    return history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry")
    args = parser.parse_args()
    train_model(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        register_model=not args.no_register,
    )


if __name__ == "__main__":
    main()
