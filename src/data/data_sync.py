from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import (
    DATASET_BUCKET,
    DATASET_INTERIM_PREFIX,
    DATASET_PROCESSED_PREFIX,
    DATASET_RAW_PREFIX,
    INTERIM_DIR,
    PROC_DIR,
    RAW_DIR,
    TRAIN_CSV,
)
from src.utils.s3_storage import download_directory, ensure_bucket, object_exists, upload_directory


def _minio_enabled() -> bool:
    return os.getenv("SKIP_MINIO", "").lower() not in {"1", "true", "yes"}


def sync_raw_from_minio(force: bool = False) -> int:
    if not _minio_enabled():
        return 0
    if not force and any(RAW_DIR.glob("*/*")):
        return 0
    ensure_bucket(DATASET_BUCKET)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return download_directory(DATASET_BUCKET, DATASET_RAW_PREFIX, RAW_DIR)


def sync_processed_from_minio(force: bool = False) -> int:
    if not _minio_enabled():
        return 0
    if not force and TRAIN_CSV.exists():
        return 0
    ensure_bucket(DATASET_BUCKET)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    count = download_directory(DATASET_BUCKET, DATASET_PROCESSED_PREFIX, PROC_DIR)
    count += download_directory(DATASET_BUCKET, DATASET_INTERIM_PREFIX, INTERIM_DIR)
    return count


def upload_processed_to_minio() -> int:
    if not _minio_enabled():
        return 0
    ensure_bucket(DATASET_BUCKET)
    count = upload_directory(PROC_DIR, DATASET_BUCKET, prefix=DATASET_PROCESSED_PREFIX)
    count += upload_directory(INTERIM_DIR, DATASET_BUCKET, prefix=DATASET_INTERIM_PREFIX)
    return count


def ensure_training_data(from_minio: bool = True) -> None:
    if from_minio and _minio_enabled():
        sync_processed_from_minio()
        if not TRAIN_CSV.exists():
            sync_raw_from_minio()


def minio_has_processed_data() -> bool:
    if not _minio_enabled():
        return False
    ensure_bucket(DATASET_BUCKET)
    return object_exists(DATASET_BUCKET, f"{DATASET_INTERIM_PREFIX}/dataset_processed.csv")
