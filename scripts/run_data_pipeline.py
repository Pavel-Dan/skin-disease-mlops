#!/usr/bin/env python3
"""End-to-end data pipeline: Kaggle -> MinIO -> preprocess -> MinIO."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data.data_sync import (
    sync_processed_from_minio,
    sync_raw_from_minio,
    upload_processed_to_minio,
)
from src.data.kaggle_fetch import download_kaggle_and_upload_minio, fetch_kaggle_to_raw, upload_raw_to_minio
from src.utils.config import DATASET_BUCKET, RAW_DIR
from src.utils.s3_storage import ensure_bucket


def _run_make_dataset(balance: bool) -> None:
    cmd = [sys.executable, "-m", "src.data.make_dataset"]
    if balance:
        cmd.append("--balance")
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_pipeline(
    from_kaggle: bool = False,
    from_minio: bool = False,
    balance: bool = True,
    purge_raw: bool = False,
    skip_upload: bool = False,
) -> dict:
    ensure_bucket(DATASET_BUCKET)

    if from_kaggle:
        print("Downloading Kaggle dataset and uploading raw data to MinIO...")
        kaggle_result = download_kaggle_and_upload_minio()
    elif from_minio:
        print("Syncing raw data from MinIO...")
        sync_raw_from_minio(force=True)
    else:
        if not any(RAW_DIR.glob("*/*")):
            raise RuntimeError(
                "No local raw data found. Use --from-kaggle or --from-minio."
            )

    if not from_kaggle and not any(RAW_DIR.glob("*/*")):
        fetch_kaggle_to_raw()

    if not skip_upload and not from_kaggle:
        print("Uploading raw data to MinIO...")
        upload_raw_to_minio()

    print("Running preprocessing...")
    _run_make_dataset(balance=balance)

    print("Uploading processed data and manifests to MinIO...")
    uploaded = upload_processed_to_minio()

    result = {
        "bucket": DATASET_BUCKET,
        "processed_uploaded": uploaded,
        "purge_raw": purge_raw,
    }

    if purge_raw:
        for child in RAW_DIR.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        print(f"Purged local raw data under {RAW_DIR}")

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle / MinIO data pipeline")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-kaggle", action="store_true", help="Download from Kaggle API")
    source.add_argument("--from-minio", action="store_true", help="Use raw data already in MinIO")
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--no-balance", action="store_false", dest="balance")
    parser.add_argument("--purge-raw", action="store_true", help="Delete local raw after preprocessing")
    parser.add_argument("--skip-upload", action="store_true", help="Skip MinIO upload steps")
    parser.add_argument("--sync-only", action="store_true", help="Only sync processed data from MinIO")
    args = parser.parse_args()

    if args.sync_only:
        count = sync_processed_from_minio(force=True)
        print(f"Synced {count} files from MinIO")
        return

    run_pipeline(
        from_kaggle=args.from_kaggle,
        from_minio=args.from_minio,
        balance=args.balance,
        purge_raw=args.purge_raw,
        skip_upload=args.skip_upload,
    )


if __name__ == "__main__":
    main()
