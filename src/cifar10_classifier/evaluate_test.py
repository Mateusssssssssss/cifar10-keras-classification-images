"""Avaliação final do checkpoint congelado no lote oficial de teste."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    top_k_accuracy_score,
)

from .data import CIFAR10_SHA256, CLASS_NAMES, load_official_test
from .evaluate import compute_metrics
from .plots import plot_class_metrics, plot_confusion


def _write_test_report(y_true, probabilities, reports, provenance):
    labels = np.arange(len(CLASS_NAMES))
    predicted = probabilities.argmax(axis=1)
    actual_indicator = y_true[:, None] == labels
    cm = confusion_matrix(y_true, predicted, labels=labels)
    report = classification_report(
        y_true, predicted, labels=labels, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    metrics = compute_metrics(y_true, predicted)
    metrics.update({
        "samples": int(y_true.size),
        "correct": int(np.count_nonzero(predicted == y_true)),
        "errors": int(np.count_nonzero(predicted != y_true)),
        "error_rate": 1 - metrics["accuracy"],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_micro": metrics["accuracy"],
        "recall_micro": metrics["accuracy"],
        "f1_micro": metrics["accuracy"],
        "top_3_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=3, labels=labels)),
        "top_5_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=5, labels=labels)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predicted, labels=labels)),
    })
    for average in ("micro", "macro", "weighted"):
        metrics[f"roc_auc_ovr_{average}"] = float(
            roc_auc_score(actual_indicator, probabilities, average=average)
        )
        metrics[f"average_precision_ovr_{average}"] = float(
            average_precision_score(actual_indicator, probabilities, average=average)
        )

    per_class = pd.DataFrame.from_dict(
        {name: report[name] for name in CLASS_NAMES}, orient="index",
    )
    per_class.index.name = "class"
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    per_class["specificity"] = tn / (tn + fp)
    per_class["roc_auc_ovr"] = roc_auc_score(actual_indicator, probabilities, average=None)
    per_class["average_precision_ovr"] = average_precision_score(
        actual_indicator, probabilities, average=None,
    )
    per_class["true_positive"] = tp
    per_class["false_positive"] = fp
    per_class["false_negative"] = fn
    per_class["true_negative"] = tn
    per_class.to_csv(reports / "test_classification_report.csv")
    pd.DataFrame(
        cm, index=pd.Index(CLASS_NAMES, name="actual"), columns=CLASS_NAMES,
    ).to_csv(reports / "test_confusion_matrix.csv")
    plot_confusion(cm, CLASS_NAMES, reports / "figures" / "confusion_matrix.png")
    plot_class_metrics(report, CLASS_NAMES, reports / "figures" / "per_class_metrics.png")
    np.savez_compressed(reports / "test_predictions.npz", labels=y_true, probabilities=probabilities)
    result = {"model": provenance, "metrics": metrics}
    (reports / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_test() -> dict:
    """Executa a avaliação final e reutiliza o relatório se já estiver disponível."""
    manifest = yaml.safe_load(Path("configs/final_test.yaml").read_text(encoding="utf-8"))
    run_name = manifest["run_name"]
    training = Path("reports") / run_name
    checkpoint = Path("models") / run_name / "best_model.keras"
    reports = Path("reports/final_test")
    summary = reports / "test_metrics.json"
    if summary.exists():
        result = json.loads(summary.read_text(encoding="utf-8"))
        if any(result["model"].get(key) != value for key, value in manifest.items()):
            raise ValueError("O teste oficial já foi registrado com outro manifesto")
        return result

    metadata = json.loads((training / "run_metadata.json").read_text(encoding="utf-8"))
    history = pd.read_csv(training / "history.csv")
    if (
        len(history) != manifest["training_epochs"]
        or metadata["best_epoch"] != manifest["best_epoch"]
        or metadata["best_epoch"] != history["val_loss"].idxmin() + 1
        or metadata["dataset_sha256"] != CIFAR10_SHA256
        or metadata["test_evaluated"]
    ):
        raise ValueError("Os registros de treino divergem do checkpoint congelado")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest["checkpoint_sha256"]:
        raise ValueError("O SHA-256 do checkpoint diverge do manifesto")

    lock = reports.parent / ".final_test.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if summary.exists():
            return evaluate_test()

        from tensorflow import keras

        model = keras.models.load_model(checkpoint, compile=False)
        images, labels = load_official_test()
        if images.shape != (10_000, 32, 32, 3) or not np.array_equal(
            np.bincount(labels), np.full(len(CLASS_NAMES), 1000),
        ):
            raise ValueError("O lote de teste não corresponde ao CIFAR-10 oficial")
        probabilities = model.predict(images, batch_size=metadata["config"]["batch_size"], verbose=0)
        provenance = {
            **manifest,
            "dataset": "CIFAR-10 official test_batch",
            "dataset_sha256": CIFAR10_SHA256,
            "model_parameters": model.count_params(),
            "selection_metric": metadata["selection_metric"],
            "validation_metrics": json.loads(
                (training / "validation_metrics.json").read_text(encoding="utf-8")
            ),
        }
        with TemporaryDirectory(dir=reports.parent, prefix=".final-test-") as temporary:
            staging = Path(temporary)
            result = _write_test_report(labels, probabilities, staging, provenance)
            staging.rename(reports)
        return result
    finally:
        lock.unlink()
