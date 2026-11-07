import io
import pickle
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from cifar10_classifier.data import CLASS_NAMES, load_development_data, stratified_indices


def test_classes():
    assert len(CLASS_NAMES) == 10


def test_stratification_disjoint_reproducible():
    y = np.repeat(np.arange(10), 20)
    train, val = stratified_indices(y, 0.2, 42)
    train_again, val_again = stratified_indices(y, 0.2, 42)
    assert set(train).isdisjoint(val)
    assert len(train) + len(val) == len(y)
    assert np.array_equal(val, val_again)
    assert np.array_equal(train, train_again)
    assert np.array_equal(np.bincount(y[val]), np.repeat(4, 10))


def test_bad_fraction():
    with pytest.raises(ValueError):
        stratified_indices(np.arange(10), 1.0, 42)


def test_development_loader_does_not_read_official_test(tmp_path, monkeypatch):
    archive = tmp_path / "cifar-10-python.tar.gz"
    labels = list(range(len(CLASS_NAMES)))
    with tarfile.open(archive, "w:gz") as batches:
        for batch_number in range(1, 6):
            pixels = np.full((len(labels), 3072), batch_number, dtype=np.uint8)
            payload = pickle.dumps({b"data": pixels, b"labels": labels})
            member = tarfile.TarInfo(f"cifar-10-batches-py/data_batch_{batch_number}")
            member.size = len(payload)
            batches.addfile(member, io.BytesIO(payload))
        payload = b"invalid pickle: this member must never be loaded"
        member = tarfile.TarInfo("cifar-10-batches-py/test_batch")
        member.size = len(payload)
        batches.addfile(member, io.BytesIO(payload))
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras.utils", SimpleNamespace(get_file=lambda **_: archive)
    )

    (x_train, y_train), (x_val, y_val) = load_development_data(0.2, 42)

    assert x_train.shape == (40, 32, 32, 3)
    assert x_val.shape == (10, 32, 32, 3)
    assert np.array_equal(np.bincount(y_train), np.full(10, 4))
    assert np.array_equal(np.bincount(y_val), np.ones(10))
