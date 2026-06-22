from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from src.data.dataset_layout import materialize_raw_layout
from src.utils.config import DATASET_BUCKET, DATASET_RAW_PREFIX, KAGGLE_DATASET, RAW_DIR
from src.utils.s3_storage import ensure_bucket, upload_directory

load_dotenv()


def _kaggle_token_path() -> Path:
    return Path.home() / ".kaggle" / "access_token"


def _ensure_kaggle_credentials() -> None:
    if os.getenv("KAGGLE_API_TOKEN"):
        return
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return
    if _kaggle_token_path().exists():
        return
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return
    raise RuntimeError(
        "Kaggle credentials required. Create a free Kaggle account, then either:\n"
        "  1. Save API token to ~/.kaggle/access_token (new format)\n"
        "  2. Or set KAGGLE_API_TOKEN in your .env file\n"
        "  3. Or use legacy ~/.kaggle/kaggle.json / KAGGLE_USERNAME + KAGGLE_KEY"
    )


def _kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        token_path = _kaggle_token_path()
        if token_path.exists():
            env["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    return env


def download_kaggle_dataset(dest_dir: Path) -> Path:
    _ensure_kaggle_credentials()
    dest_dir.mkdir(parents=True, exist_ok=True)
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        candidate = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
        kaggle_bin = str(candidate) if candidate.exists() else "kaggle"
    cmd = [
        kaggle_bin,
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(dest_dir),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True, env=_kaggle_env())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not installed. Run: pip install kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Kaggle download failed for {KAGGLE_DATASET}. "
            "Accept the dataset rules on kaggle.com and verify credentials."
        ) from exc

    zip_files = list(dest_dir.glob("*.zip"))
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as archive:
            archive.extractall(dest_dir)
        zf.unlink(missing_ok=True)
    return dest_dir


def fetch_kaggle_to_raw(raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or RAW_DIR
    with tempfile.TemporaryDirectory(prefix="kaggle_skin_") as tmp:
        download_dir = Path(tmp)
        download_kaggle_dataset(download_dir)
        materialize_raw_layout(download_dir, raw_dir=raw_dir)
    return raw_dir


def upload_raw_to_minio(raw_dir: Path | None = None) -> int:
    raw_dir = raw_dir or RAW_DIR
    ensure_bucket(DATASET_BUCKET)
    return upload_directory(raw_dir, DATASET_BUCKET, prefix=DATASET_RAW_PREFIX)


def download_kaggle_and_upload_minio(raw_dir: Path | None = None, purge_temp: bool = True) -> dict:
    raw_dir = raw_dir or RAW_DIR
    fetch_kaggle_to_raw(raw_dir=raw_dir)
    uploaded = upload_raw_to_minio(raw_dir=raw_dir)
    return {"raw_dir": str(raw_dir), "uploaded_files": uploaded, "bucket": DATASET_BUCKET}
