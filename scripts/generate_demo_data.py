"""Generate a minimal synthetic dataset for CI and local testing without the full raw corpus."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.utils.config import CLASSES, INTERIM_DIR, PROC_DIR, RAW_DIR, SEED

random.seed(SEED)


def _write_images(base: Path, label: str, count: int, prefix: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    out_dir = base / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        color = tuple(random.randint(40, 220) for _ in range(3))
        img = Image.new("RGB", (224, 224), color)
        path = out_dir / f"{prefix}_{label}_{i:03d}.jpg"
        img.save(path, format="JPEG", quality=90)
        rows.append((str(path), label))
    return rows


def generate_raw(n_per_class: int = 20) -> None:
    for label in CLASSES:
        _write_images(RAW_DIR, label, n_per_class, "raw")


def generate_processed_manifest(n_per_split: int = 12) -> Path:
    rows: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        for label in CLASSES:
            rows.extend(_write_images(PROC_DIR / split, label, n_per_split, split))
    csv_path = INTERIM_DIR / "dataset_processed.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-per-class", type=int, default=20)
    parser.add_argument("--processed-per-split", type=int, default=12)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--processed-only", action="store_true")
    args = parser.parse_args()

    if not args.processed_only:
        generate_raw(args.raw_per_class)
        print(f"Demo raw images written under {RAW_DIR}")

    if not args.raw_only:
        path = generate_processed_manifest(args.processed_per_class)
        print(f"Demo processed manifest written to {path}")


if __name__ == "__main__":
    main()
