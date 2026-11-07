"""Treinamento reprodutível, exclusivamente com validação."""
import json
import random
import shutil
from importlib.metadata import version
from platform import python_version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import CIFAR10_SHA256, load_development_data
from .model import build_cnn
from .plots import plot_training


def _resume_training(keras, cfg, split_path, checkpoint):
    parent = cfg["resume_from"]
    parent_reports = Path("reports") / parent
    parent_metadata = json.loads((parent_reports / "run_metadata.json").read_text(encoding="utf-8"))
    parent_history = pd.read_csv(parent_reports / "history.csv")
    completed = len(parent_history)
    allowed_changes = {"epochs", "patience", "run_name", "resume_from"}
    comparable = {key: value for key, value in cfg.items() if key not in allowed_changes}
    previous_cfg = {
        key: value for key, value in parent_metadata["config"].items() if key not in allowed_changes
    }
    if comparable != previous_cfg:
        raise ValueError("A continuação requer a mesma arquitetura e configuração de treino")
    if parent_metadata["dataset_sha256"] != CIFAR10_SHA256 or parent_metadata["test_evaluated"]:
        raise ValueError("A continuação requer o mesmo conjunto oficial de desenvolvimento")
    if not completed == parent_metadata["best_epoch"] < cfg["epochs"]:
        raise ValueError("A continuação requer um checkpoint da última época e um limite maior")

    with (
        np.load(Path("data/processed") / parent / "split_indices.npz") as previous,
        np.load(split_path) as current,
    ):
        for key in ("train_idx", "val_idx"):
            np.testing.assert_array_equal(current[key], previous[key])

    parent_checkpoint = Path("models") / parent / "best_model.keras"
    model = keras.models.load_model(parent_checkpoint)
    shutil.copyfile(parent_checkpoint, checkpoint)
    return model, parent_history


def train(config_path: str = "configs/train.yaml") -> dict:
    import tensorflow as tf
    from tensorflow import keras

    cfg = load_config(config_path)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    keras.utils.set_random_seed(cfg["seed"])
    try:
        tf.config.experimental.enable_op_determinism()
    except (RuntimeError, NotImplementedError):
        pass
    run_name = cfg.get("run_name", "")
    models = Path("models") / run_name
    reports = Path("reports") / run_name
    split_path = Path("data/processed") / run_name / "split_indices.npz"
    models.mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_val, y_val) = load_development_data(
        cfg["validation_fraction"], cfg["seed"], split_path
    )
    checkpoint = models / "best_model.keras"
    prior_history = pd.DataFrame()
    best_loss = None
    if cfg.get("resume_from"):
        model, prior_history = _resume_training(keras, cfg, split_path, checkpoint)
        best_loss = float(prior_history["val_loss"].min())
    else:
        filters_per_block = tuple(cfg.get("filters_per_block", (32, 64, 128)))
        model = build_cnn(
            cfg["dropout"], cfg["learning_rate"], cfg["weight_decay"],
            filters_per_block=filters_per_block,
            architecture=cfg.get("architecture", "baseline"),
        )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, initial_value_threshold=best_loss
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg["reduce_lr_patience"]
        ),
        keras.callbacks.CSVLogger(reports / "epoch_log.csv"),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg["batch_size"], initial_epoch=len(prior_history), epochs=cfg["epochs"],
        callbacks=callbacks, verbose=2,
    )
    full_history = pd.concat([prior_history, pd.DataFrame(history.history)], ignore_index=True)
    full_history.to_csv(reports / "history.csv", index=False)
    plot_training(full_history, reports / "figures" / "training_curves.png")
    metadata = {
        "config": cfg,
        "dataset_sha256": CIFAR10_SHA256,
        "model_parameters": model.count_params(),
        "versions": {
            "python": python_version(),
            "tensorflow": tf.__version__,
            **{name: version(name) for name in ("keras", "numpy", "scikit-learn")},
        },
        "train_count": len(y_train), "validation_count": len(y_val),
        "selection_metric": "val_loss",
        "best_epoch": int(full_history["val_loss"].idxmin() + 1),
        "resumed_at_epoch": len(prior_history),
        "test_evaluated": False,
    }
    (reports / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
