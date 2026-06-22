#!/usr/bin/env bash
set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
BUCKET="${MLFLOW_BUCKET:-mlflow}"
DATA_BUCKET="${DATASET_BUCKET:-skin-datasets}"

if command -v mc >/dev/null 2>&1; then
  mc alias set local "${MINIO_ENDPOINT}" "${MINIO_USER}" "${MINIO_PASSWORD}"
  mc mb "local/${BUCKET}" --ignore-existing
  mc mb "local/${DATA_BUCKET}" --ignore-existing
  echo "Buckets '${BUCKET}' and '${DATA_BUCKET}' are ready."
else
  python - <<'PY'
import os
import boto3
from botocore.client import Config

endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
user = os.getenv("MINIO_ROOT_USER", "minioadmin")
password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
buckets = [
    os.getenv("MLFLOW_BUCKET", "mlflow"),
    os.getenv("DATASET_BUCKET", "skin-datasets"),
]

client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=user,
    aws_secret_access_key=password,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)
existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
for bucket in buckets:
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)
    print(f"Bucket '{bucket}' is ready.")
PY
fi
