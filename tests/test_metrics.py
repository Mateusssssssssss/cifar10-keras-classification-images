import numpy as np
from cifar10_classifier.evaluate import compute_metrics


def test_perfect_predictions():
    labels = np.arange(10)
    metrics = compute_metrics(labels, labels)
    assert all(value == 1.0 for value in metrics.values())


def test_imperfect_predictions():
    actual = np.arange(10)
    predicted = np.zeros(10, dtype=int)
    metrics = compute_metrics(actual, predicted)
    assert metrics["accuracy"] == 0.1
    assert 0 <= metrics["f1_macro"] < 1
