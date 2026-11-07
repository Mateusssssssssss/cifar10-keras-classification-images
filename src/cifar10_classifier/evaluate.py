"""Métricas de inferência para treino e validação."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from .config import load_config
from .data import CLASS_NAMES, load_development_data
from .plots import plot_class_metrics, plot_confusion


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de classificação multiclasse para todas as classes."""
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def evaluate_validation(config_path: str = "configs/train.yaml") -> dict:
    from tensorflow import keras

    cfg = load_config(config_path)
    run_name = cfg.get("run_name", "")
    reports = Path("reports") / run_name
    checkpoint = Path("models") / run_name / "best_model.keras"
    if not checkpoint.exists():
        raise FileNotFoundError("Execute scripts/train.py primeiro: checkpoint não encontrado")
    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    if cfg != metadata["config"]:
        raise ValueError("Use a mesma configuração usada no treino do checkpoint")
    (x_train, y_train), (x_val, y_val) = load_development_data(
        cfg["validation_fraction"], cfg["seed"]
    )
    model = keras.models.load_model(checkpoint)
    train_probabilities = model.predict(x_train, batch_size=cfg["batch_size"], verbose=0)
    train_metrics = compute_metrics(y_train, np.argmax(train_probabilities, axis=1))
    probabilities = model.predict(x_val, batch_size=cfg["batch_size"], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    metrics = compute_metrics(y_val, predictions)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )
    (reports / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(
        y_val, predictions, labels=list(range(10)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(reports / "validation_classification_report.csv")
    cm = confusion_matrix(y_val, predictions, labels=list(range(10)))
    plot_confusion(cm, CLASS_NAMES, figures / "confusion_matrix.png")
    plot_class_metrics(report, CLASS_NAMES, figures / "per_class_metrics.png")
    wrong = np.flatnonzero(predictions != y_val)
    pd.DataFrame({
        "validation_index": wrong,
        "actual": [CLASS_NAMES[i] for i in y_val[wrong]],
        "predicted": [CLASS_NAMES[i] for i in predictions[wrong]],
        "confidence": probabilities[wrong].max(axis=1),
    }).to_csv(reports / "validation_errors.csv", index=False)
    return metrics
