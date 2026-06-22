from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitoring.drift_report import generate_drift_report
from src.training.evaluate import evaluate
from src.training.train import train_model
from src.utils.config import DRIFT_REPORT_DIR, INTERIM_DIR, MODEL_NAME, TRAIN_CSV


def _latest_production_f1() -> float:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        production = [v for v in versions if v.current_stage == "Production"]
        if not production:
            return -1.0
        latest = max(production, key=lambda v: int(v.version))
        return float(latest.tags.get("val_f1_macro", 0) or 0)
    except Exception:
        metrics_path = INTERIM_DIR / "test_metrics.json"
        if metrics_path.exists():
            return float(json.loads(metrics_path.read_text(encoding="utf-8")).get("f1_macro", -1))
        return -1.0


def run_retrain(
    force: bool = False,
    csv_path: Path | None = None,
    epochs: int | None = None,
    skip_drift_check: bool = False,
) -> dict:
    drift = {"drift_detected": force}
    if not force and not skip_drift_check:
        drift = generate_drift_report()
        if not drift.get("drift_detected"):
            return {"status": "skipped", "reason": "No drift detected", "drift": drift}

    from src.data.data_sync import ensure_training_data

    ensure_training_data(from_minio=True)
    history = train_model(csv_path=csv_path, epochs=epochs or 3, register_model=True)
    new_f1 = float(history.get("best_val_f1_macro", 0))
    previous_f1 = _latest_production_f1()
    promoted = new_f1 >= previous_f1

    result = {
        "status": "completed",
        "promoted": promoted,
        "new_val_f1_macro": new_f1,
        "previous_val_f1_macro": previous_f1,
        "drift": drift,
    }
    out_path = DRIFT_REPORT_DIR / "retrain_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-drift-check", action="store_true")
    parser.add_argument("--csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate(csv_path=args.csv)
        return

    run_retrain(
        force=args.force,
        csv_path=args.csv,
        epochs=args.epochs,
        skip_drift_check=args.skip_drift_check,
    )


if __name__ == "__main__":
    main()
