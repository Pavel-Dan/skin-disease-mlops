from __future__ import annotations

import re
import shutil
from pathlib import Path

from src.utils.config import CLASS_FOLDER_RULES, CLASS_DIRS, CLASSES, IMG_EXTS, RAW_DIR


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def match_label_from_folder(folder_name: str) -> str | None:
    normalized = _normalize_name(folder_name)
    for patterns, label in CLASS_FOLDER_RULES:
        if any(pattern in normalized for pattern in patterns):
            return label
    return None


def find_dataset_root(search_dir: Path) -> Path:
    if not search_dir.exists():
        raise FileNotFoundError(f"Search directory not found: {search_dir}")

    for name in ("IMG_CLASSES", "img_classes", "dataset", "data"):
        candidate = search_dir / name
        if candidate.is_dir():
            return candidate

    image_dirs = [p for p in search_dir.rglob("*") if p.is_dir() and any(p.glob("*.*"))]
    class_like = [p for p in image_dirs if match_label_from_folder(p.name)]
    if class_like:
        return class_like[0].parent

    return search_dir


def discover_class_directories(raw_dir: Path | None = None) -> dict[Path, str]:
    raw_dir = raw_dir or RAW_DIR
    by_label: dict[str, Path] = {}

    for label in CLASSES:
        path = raw_dir / label
        if path.is_dir() and any(path.rglob("*")):
            by_label[label] = path

    root = find_dataset_root(raw_dir)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        label = match_label_from_folder(path.name)
        if label is None:
            continue
        if not any(child.suffix.lower() in IMG_EXTS for child in path.rglob("*") if child.is_file()):
            continue
        by_label.setdefault(label, path)

    for folder_name, label in CLASS_DIRS.items():
        for base in (raw_dir, root):
            path = base / folder_name
            if path.is_dir():
                by_label.setdefault(label, path)
                break

    return {path: label for label, path in by_label.items()}


def materialize_raw_layout(source_root: Path, raw_dir: Path | None = None) -> dict[str, Path]:
    """Copy class folders into data/raw/{label}/ for a stable layout."""
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = find_dataset_root(source_root)
    class_dirs = discover_class_directories(dataset_root)

    if not class_dirs:
        class_dirs = discover_class_directories(source_root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found under {source_root}. "
            "Check Kaggle download or MinIO sync."
        )

    layout: dict[str, Path] = {}
    for src, label in class_dirs.items():
        target = raw_dir / label
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for image in src.rglob("*"):
            if image.is_file() and image.suffix.lower() in IMG_EXTS:
                dest = target / image.name
                if dest.exists():
                    dest = target / f"{src.name}_{image.name}"
                shutil.copy2(image, dest)
        layout[label] = target
    return layout
