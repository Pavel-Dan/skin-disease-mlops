from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.utils.config import CLASSES, DRIFT_REPORT_DIR, DRIFT_THRESHOLD, INTERIM_DIR, LOG_PREDICTIONS_PATH


def _reference_distribution() -> pd.Series:
    counts_path = INTERIM_DIR / "class_counts.json"
    if counts_path.exists():
        counts = json.loads(counts_path.read_text(encoding="utf-8")).get("test", {})
        return pd.Series({label: counts.get(label, 0) for label in CLASSES}, dtype=float)

    csv_path = INTERIM_DIR / "dataset_processed.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["split"] = df["filepath"].str.replace("\\", "/").str.extract(r"/(train|val|test)/")[0]
        test_df = df[df["split"] == "test"]
        if not test_df.empty:
            return test_df["label"].value_counts(normalize=True)

    uniform = 1.0 / len(CLASSES)
    return pd.Series({label: uniform for label in CLASSES}, dtype=float)


def _current_distribution() -> pd.Series:
    if not LOG_PREDICTIONS_PATH.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(LOG_PREDICTIONS_PATH)
    if df.empty:
        return pd.Series(dtype=float)
    return df["predicted_class"].value_counts(normalize=True)


def _population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    aligned_ref = pd.Series({label: reference.get(label, 0.0) for label in CLASSES}, dtype=float)
    aligned_cur = pd.Series({label: current.get(label, 0.0) for label in CLASSES}, dtype=float)
    aligned_ref = aligned_ref / aligned_ref.sum() if aligned_ref.sum() else aligned_ref
    aligned_cur = aligned_cur / aligned_cur.sum() if aligned_cur.sum() else aligned_cur
    psi = 0.0
    for label in CLASSES:
        expected = max(float(aligned_ref.get(label, 0.0)), 1e-6)
        actual = max(float(aligned_cur.get(label, 0.0)), 1e-6)
        psi += (actual - expected) * math.log(actual / expected)
    return float(abs(psi))


def generate_drift_report(threshold: float | None = None) -> dict:
    threshold = threshold if threshold is not None else DRIFT_THRESHOLD
    reference = _reference_distribution()
    current = _current_distribution()

    if current.empty:
        result = {
            "drift_detected": False,
            "drift_score": 0.0,
            "message": "No prediction logs available yet.",
        }
    else:
        drift_score = _population_stability_index(reference, current)
        result = {
            "drift_detected": drift_score >= threshold,
            "drift_score": drift_score,
            "message": "Drift score computed from prediction class distribution (PSI-like).",
            "reference_distribution": reference.to_dict(),
            "current_distribution": current.to_dict(),
        }

        DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = DRIFT_REPORT_DIR / "drift.html"
        rows = []
        for label in CLASSES:
            rows.append(
                {
                    "class": label,
                    "reference": float(reference.get(label, 0.0)),
                    "current": float(current.get(label, 0.0)),
                }
            )
        table = pd.DataFrame(rows)
        html_path.write_text(
            table.to_html(index=False, title="Prediction Drift Report"),
            encoding="utf-8",
        )

        try:
            from evidently import Report
            from evidently.metrics import ValueDrift

            ref_df = pd.DataFrame({"predicted_class": reference.index.repeat(reference.astype(int).clip(lower=1))})
            cur_df = pd.DataFrame({"predicted_class": current.index.repeat(current.mul(100).astype(int).clip(lower=1))})
            report = Report(metrics=[ValueDrift(column="predicted_class")])
            report.run(reference_data=ref_df, current_data=cur_df)
            report.save_html(str(DRIFT_REPORT_DIR / "drift_evidently.html"))
            result["evidently_report"] = "monitoring/reports/drift_evidently.html"
        except Exception as exc:
            result["evidently_report_error"] = str(exc)

    score_path = DRIFT_REPORT_DIR / "drift_score.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"reference_distribution", "current_distribution"}}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    args = parser.parse_args()
    generate_drift_report(threshold=args.threshold)


if __name__ == "__main__":
    main()
