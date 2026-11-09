"""Carregamento do CIFAR-10 e divisão reproduzível de treino/validação."""
import pickle
import tarfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_URL = f"https://cave.cs.toronto.edu/kriz/{CIFAR10_ARCHIVE}"
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def stratified_indices(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Divide índices com estratificação; nunca inclui dados oficiais de teste."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    y = np.asarray(y).reshape(-1)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=fraction, random_state=seed, stratify=y
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def _load_batches(batch_names):
    from tensorflow.keras.utils import get_file

    archive = get_file(
        fname=CIFAR10_ARCHIVE,
        origin=CIFAR10_URL,
        file_hash=CIFAR10_SHA256,
        cache_dir=str(Path("data").resolve()),
        cache_subdir="raw",
    )
    images, labels = [], []
    with tarfile.open(archive, "r:gz") as batches:
        for name in batch_names:
            batch_path = f"cifar-10-batches-py/{name}"
            with batches.extractfile(batch_path) as handle:
                batch = pickle.load(handle, encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(batch[b"labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


def load_official_test():
    """Lê somente o lote reservado à avaliação final."""
    return _load_batches(("test_batch",))


def load_development_data(fraction: float, seed: int, save_indices: str | Path | None = None):
    """Lê exclusivamente os cinco lotes oficiais de desenvolvimento."""
    x, y = _load_batches(f"data_batch_{number}" for number in range(1, 6))
    train_idx, val_idx = stratified_indices(y, fraction, seed)
    if save_indices is not None:
        target = Path(save_indices)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, train_idx=train_idx, val_idx=val_idx)
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
