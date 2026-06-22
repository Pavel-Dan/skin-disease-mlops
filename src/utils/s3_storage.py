from __future__ import annotations

import mimetypes
import os
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.utils.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    DATASET_BUCKET,
    MLFLOW_S3_ENDPOINT_URL,
)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def ensure_bucket(bucket: str) -> None:
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    client = get_s3_client()
    extra = {}
    content_type, _ = mimetypes.guess_type(str(local_path))
    if content_type:
        extra["ContentType"] = content_type
    if extra:
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
    else:
        client.upload_file(str(local_path), bucket, key)


def download_file(bucket: str, key: str, local_path: Path) -> None:
    client = get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(local_path))


def upload_directory(local_dir: Path, bucket: str, prefix: str = "") -> int:
    if not local_dir.exists():
        return 0
    count = 0
    prefix = prefix.strip("/")
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            upload_file(path, bucket, key)
            count += 1
    return count


def download_directory(bucket: str, prefix: str, local_dir: Path) -> int:
    client = get_s3_client()
    prefix = prefix.strip("/")
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) + 1 :] if prefix and key.startswith(prefix + "/") else key
            if not rel:
                continue
            target = local_dir / rel
            download_file(bucket, key, target)
            count += 1
    return count


def object_exists(bucket: str, key: str) -> bool:
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def list_keys(bucket: str, prefix: str = "") -> list[str]:
    client = get_s3_client()
    prefix = prefix.strip("/")
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
