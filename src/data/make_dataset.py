import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Augmentations pour l'oversampling
import albumentations as A
from albumentations.pytorch import ToTensorV2  # (utile plus tard)

from src.utils.config import (
    RAW_DIR, PROC_DIR, INTERIM_DIR, IMG_SIZE, SPLIT, SEED, CLASSES,
    CLASS_DIRS, IMG_EXTS, BALANCING
)

random.seed(SEED)
np.random.seed(SEED)

AUG_PIPELINE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
])

def _list_images(d: Path) -> List[Path]:
    return [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]

def _validate_and_load(path: Path) -> Image.Image | None:
    try:
        Image.open(path).verify()
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def _resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BILINEAR)

def _albumentations_to_pil(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    augmented = AUG_PIPELINE(image=arr)["image"]
    return Image.fromarray(augmented)

def build_manifest() -> list[tuple[str, str]]:
    """Return (filepath, label) pairs from discovered class folders."""
    from src.data.dataset_layout import discover_class_directories

    pairs = []
    class_dirs = discover_class_directories(RAW_DIR)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in {RAW_DIR}. "
            "Run: python scripts/run_data_pipeline.py --from-minio"
        )
    for class_dir, clean_label in class_dirs.items():
        for p in _list_images(class_dir):
            pairs.append((str(p), clean_label))
    random.shuffle(pairs)
    return pairs

def split_indices_by_label(manifest: list[tuple[str, str]]):
    """Stratified split: on découpe par classe pour garder les proportions dans chaque split."""
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, (_, y) in enumerate(manifest):
        by_label[y].append(i)

    splits = {"train": [], "val": [], "test": []}
    for y, idxs in by_label.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_tr = int(SPLIT["train"] * n)
        n_val = int(SPLIT["val"] * n)
        splits["train"].extend(idxs[:n_tr])
        splits["val"].extend(idxs[n_tr:n_tr + n_val])
        splits["test"].extend(idxs[n_tr + n_val:])
    # on shuffle chaque split
    for k in splits: random.shuffle(splits[k])
    return splits

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for label in CLASSES:
            (PROC_DIR / split / label).mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def save_csv(rows: list[list[str]], dest_csv: Path, header=("filepath","label")):
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)

def compute_class_weights(counts: Dict[str, int]) -> Dict[str, float]:
    # inverse frequency normalisée
    total = sum(counts.values())
    inv = {k: (1.0 / v if v > 0 else 0.0) for k, v in counts.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}

def process_and_save(src_path: Path, dest_dir: Path, augment=False):
    img = _validate_and_load(src_path)
    if img is None:
        return None
    if augment:
        img = _albumentations_to_pil(img)
    img = _resize(img, IMG_SIZE)
    dest_path = dest_dir / src_path.name
    # éviter collisions de noms quand on duplique -> préfixe
    if dest_path.exists():
        dest_path = dest_dir / f"aug_{random.randrange(10_000_000)}_{src_path.name}"
    img.save(dest_path, format="JPEG", quality=95)
    return dest_path

def main(apply_balance: bool):
    ensure_dirs()
    manifest = build_manifest()
    save_csv(manifest, INTERIM_DIR / "manifest_raw.csv")  # pour traçabilité

    splits = split_indices_by_label(manifest)

    final_rows = []
    split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    # 1) VAL / TEST (jamais équilibrés)
    for sp in ["val", "test"]:
        for i in tqdm(splits[sp], desc=f"Processing {sp}"):
            src, label = manifest[i]
            dest_dir = PROC_DIR / sp / label
            saved = process_and_save(Path(src), dest_dir, augment=False)
            if saved:
                final_rows.append([str(saved), label])
                split_counts[sp][label] += 1

    # 2) TRAIN (éventuellement équilibré)
    #    - on traite d'abord une passe "simple"
    train_rows_by_label: Dict[str, List[list[str]]] = defaultdict(list)
    for i in tqdm(splits["train"], desc="Processing train (base)"):
        src, label = manifest[i]
        dest_dir = PROC_DIR / "train" / label
        saved = process_and_save(Path(src), dest_dir, augment=False)
        if saved:
            row = [str(saved), label]
            train_rows_by_label[label].append(row)

    # Équilibrage si demandé
    if apply_balance and BALANCING["strategy"] == "oversample":
        present_labels = [y for y, rows in train_rows_by_label.items() if rows]
        missing = [y for y in CLASSES if y not in present_labels]
        if missing:
            print(f"⚠️  Classes absentes du train (ignorées pour l'équilibrage): {', '.join(missing)}")
        if not present_labels:
            raise RuntimeError("Aucune image d'entraînement valide après prétraitement.")

        current_counts = {y: len(train_rows_by_label[y]) for y in present_labels}
        if BALANCING["target"] == "max":
            target = max(current_counts.values())
        else:
            target = int(BALANCING["target"])

        for y in present_labels:
            need = target - len(train_rows_by_label[y])
            if need <= 0:
                continue
            base_files = [Path(r[0]) for r in train_rows_by_label[y]]
            if not base_files:
                continue
            for _ in tqdm(range(need), desc=f"Oversample {y}", leave=False):
                src_choice = random.choice(base_files)
                dest_dir = PROC_DIR / "train" / y
                saved = process_and_save(src_choice, dest_dir, augment=BALANCING["augment"])
                if saved:
                    train_rows_by_label[y].append([str(saved), y])

    # consolider le TRAIN
    for y in CLASSES:
        for row in train_rows_by_label[y]:
            final_rows.append(row)
            split_counts["train"][y] += 1

    # CSV final
    save_csv(final_rows, INTERIM_DIR / "dataset_processed.csv")

    # Sauver des stats + poids de classes (pour CrossEntropyLoss)
    totals = {y: split_counts["train"][y] + split_counts["val"][y] + split_counts["test"][y] for y in CLASSES}
    class_weights = compute_class_weights(split_counts["train"])
    (INTERIM_DIR / "class_counts.json").write_text(json.dumps({
        "train": split_counts["train"], "val": split_counts["val"], "test": split_counts["test"],
        "totals": totals
    }, default=int, indent=2))
    (INTERIM_DIR / "class_weights_train.json").write_text(json.dumps(class_weights, indent=2))

    print("Pretraitement termine.")
    print(f"- CSV: {INTERIM_DIR / 'dataset_processed.csv'}")
    print(f"- Comptes: {INTERIM_DIR / 'class_counts.json'}")
    print(f"- Poids de classes (train): {INTERIM_DIR / 'class_weights_train.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", action="store_true", help="Équilibrer le TRAIN par oversampling")
    args = parser.parse_args()
    main(apply_balance=args.balance)
