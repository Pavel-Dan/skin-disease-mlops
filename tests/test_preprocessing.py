import random

from src.data.make_dataset import split_indices_by_label
from src.utils.config import SEED


def test_stratified_split_preserves_labels() -> None:
    manifest = [(f"img_{i}.jpg", label) for label in ("eczema", "melanoma", "bcc") for i in range(30)]
    splits = split_indices_by_label(manifest)

    for split_name, indices in splits.items():
        labels = {manifest[i][1] for i in indices}
        assert labels.issubset({"eczema", "melanoma", "bcc"})
        assert len(indices) > 0


def test_no_train_indices_in_val_or_test() -> None:
    manifest = [(f"img_{i}.jpg", label) for label in ("eczema", "melanoma") for i in range(50)]
    splits = split_indices_by_label(manifest)
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_split_is_deterministic_with_seed() -> None:
    manifest = [(f"img_{i}.jpg", "eczema") for i in range(20)]
    random.seed(SEED)
    first = split_indices_by_label(manifest)
    random.seed(SEED)
    second = split_indices_by_label(manifest)
    assert first == second
