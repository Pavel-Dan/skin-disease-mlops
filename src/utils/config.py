import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = Path("models")

# Kaggle dataset: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "ismailpromus/skin-diseases-image-dataset")

# Canonical labels — 6-class scope (subset of Kaggle dataset, already downloaded)
CLASSES = [
    "eczema",
    "melanoma",
    "ad",
    "bcc",
    "bkl",
    "pso",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Folder name patterns (lowercase substrings) -> canonical label
CLASS_FOLDER_RULES: list[tuple[tuple[str, ...], str]] = [
    (("eczema",), "eczema"),
    (("melanocytic nevi",), "nv"),
    (("melanoma",), "melanoma"),
    (("atopic dermatitis", "atopic"), "ad"),
    (("basal cell", "bcc"), "bcc"),
    (("benign keratosis", "bkl"), "bkl"),
    (("psoriasis", "lichen planus"), "pso"),
    (("seborrheic",), "seborrheic"),
    (("tinea", "ringworm", "candidiasis", "fungal"), "tinea"),
    (("warts", "molluscum", "viral infections"), "warts"),
]

# Legacy fixed mapping (original 6-class repo) — used as fallback
CLASS_DIRS = {
    "1. Eczema 1677": "eczema",
    "2. Melanoma 15.75k": "melanoma",
    "4. Basal Cell Carcinoma (BCC) 3323": "bcc",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "bkl",
    "3. Atopic Dermatitis - 1.25k": "ad",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "pso",
}

# Preprocessing
IMG_SIZE = (300, 300)  # EfficientNet-B3 default resolution
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BALANCING = {
    "strategy": "oversample",
    "target": "max",
    "augment": True,
}

# MLflow & MinIO
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ROOT_USER", "minioadmin"))
AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
)
MLFLOW_BUCKET = os.getenv("MLFLOW_BUCKET", "mlflow")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "skin-datasets")
DATASET_RAW_PREFIX = os.getenv("DATASET_RAW_PREFIX", "raw")
DATASET_PROCESSED_PREFIX = os.getenv("DATASET_PROCESSED_PREFIX", "processed")
DATASET_INTERIM_PREFIX = os.getenv("DATASET_INTERIM_PREFIX", "interim")

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "skin-classifier")
MODEL_ARCH = os.getenv("MODEL_ARCH", "efficientnet_b3")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pt"))
LOG_PREDICTIONS_PATH = Path(os.getenv("LOG_PREDICTIONS_PATH", "data/logs/predictions.csv"))

# Training defaults
TRAIN_CSV = INTERIM_DIR / "dataset_processed.csv"
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "15"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("TRAIN_LR", "3e-4"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "4"))

# Monitoring
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
DRIFT_REPORT_DIR = Path("monitoring/reports")

MEDICAL_DISCLAIMER = (
    "This tool is for decision support only and does not replace professional medical diagnosis."
)
