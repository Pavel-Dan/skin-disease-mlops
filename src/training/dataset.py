from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.training.model import build_model, get_model_transforms
from src.utils.config import CLASS_TO_IDX, CLASSES, SEED, TRAIN_CSV


def infer_split(filepath: str) -> str:
    normalized = filepath.replace("\\", "/")
    for split in ("train", "val", "test"):
        if f"/{split}/" in normalized or normalized.startswith(f"{split}/"):
            return split
    parts = normalized.split("/")
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def get_transforms(train: bool = False) -> transforms.Compose:
    return get_model_transforms(train=train)


class SkinLesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = get_transforms(train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = CLASS_TO_IDX[row["label"]]
        return self.transform(image), label


def load_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or TRAIN_CSV
    df = pd.read_csv(path)
    if "split" not in df.columns:
        df["split"] = df["filepath"].apply(infer_split)
    return df


def create_dataloaders(
    csv_path: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    df = load_dataframe(csv_path)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if part.empty:
            continue
        loaders[split] = DataLoader(
            SkinLesionDataset(part, train=(split == "train")),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(SEED),
        )
    return loaders
