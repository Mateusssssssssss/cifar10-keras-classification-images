"""Gráficos de treinamento e validação."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training(history, destination: str | Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    for axis, train, val, title in zip(
        axes, ("loss", "accuracy"), ("val_loss", "val_accuracy"),
        ("Loss", "Accuracy"), strict=True,
    ):
        axis.plot(epochs, history[train], label="Train")
        axis.plot(epochs, history[val], label="Validation")
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names, destination: str | Path):
    from sklearn.metrics import ConfusionMatrixDisplay

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_class_metrics(report, class_names, destination: str | Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(class_names))
    for offset, label in zip((-0.25, 0.0, 0.25), ("precision", "recall", "f1-score"), strict=True):
        values = [report[name][label] for name in class_names]
        bars = ax.bar(positions + offset, values, 0.24, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(positions, class_names, rotation=35)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
