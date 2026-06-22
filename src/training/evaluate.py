from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.training.dataset import create_dataloaders
from src.training.model import build_model
from src.training.train import collect_predictions, compute_metrics
from src.utils.config import INTERIM_DIR, MODEL_PATH, TRAIN_CSV


def evaluate(csv_path: Path | None = None, checkpoint_path: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders(csv_path=csv_path or TRAIN_CSV)
    if "test" not in loaders:
        raise RuntimeError("Test split is required for evaluation.")

    checkpoint = torch.load(checkpoint_path or MODEL_PATH, map_location=device, weights_only=False)
    model = build_model(pretrained=False, arch=checkpoint.get("model_arch")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    out_path = INTERIM_DIR / "evaluation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    evaluate(csv_path=args.csv, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
