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


def _build_model(cfg):
    return build_cnn(
        cfg["dropout"], cfg["learning_rate"], cfg["weight_decay"],
        filters_per_block=tuple(cfg.get("filters_per_block", (32, 64, 128))),
        architecture=cfg.get("architecture", "baseline"),
        bicubic_size=cfg.get("bicubic_size"),
    )


def _transfer_model_state(source, target):
    target.set_weights(source.get_weights())
    target.optimizer.build(target.trainable_variables)
    target.optimizer.set_weights([variable.numpy() for variable in source.optimizer.variables])


def _resume_training(keras, cfg, split_path, checkpoint):
    fine_tune = bool(cfg.get("fine_tune_from"))
    parent = cfg.get("fine_tune_from") or cfg["resume_from"]
    parent_reports = Path("reports") / parent
    parent_metadata = json.loads((parent_reports / "run_metadata.json").read_text(encoding="utf-8"))
    parent_history = pd.read_csv(parent_reports / "history.csv")
    start_epoch = parent_metadata["best_epoch"]
    allowed_changes = {
        "epochs", "patience", "run_name", "resume_from", "fine_tune_from", "photometric_augmentation",
    }
    if fine_tune:
        allowed_changes.add("bicubic_size")
    comparable = {key: value for key, value in cfg.items() if key not in allowed_changes}
    previous_cfg = {
        key: value for key, value in parent_metadata["config"].items() if key not in allowed_changes
    }
    if comparable != previous_cfg:
        raise ValueError("A continuação requer a mesma arquitetura e configuração de treino")
    if parent_metadata["dataset_sha256"] != CIFAR10_SHA256 or parent_metadata["test_evaluated"]:
        raise ValueError("A continuação requer o mesmo conjunto oficial de desenvolvimento")
    if not 0 < start_epoch <= len(parent_history) or start_epoch >= cfg["epochs"]:
        raise ValueError("A época do checkpoint precisa ser anterior ao novo limite")
    if not fine_tune and start_epoch != len(parent_history):
        raise ValueError("A continuação requer um checkpoint da última época")

    with (
        np.load(Path("data/processed") / parent / "split_indices.npz") as previous,
        np.load(split_path) as current,
    ):
        for key in ("train_idx", "val_idx"):
            np.testing.assert_array_equal(current[key], previous[key])

    parent_checkpoint = Path("models") / parent / "best_model.keras"
    model = keras.models.load_model(parent_checkpoint)
    if cfg.get("bicubic_size") != parent_metadata["config"].get("bicubic_size"):
        resized_model = _build_model(cfg)
        _transfer_model_state(model, resized_model)
        model = resized_model
    if not fine_tune:
        shutil.copyfile(parent_checkpoint, checkpoint)
    return model, parent_history.iloc[:start_epoch]


def _fine_tune_dataset(tf, keras, images, labels, cfg):
    factors = cfg.get("photometric_augmentation", {})
    brightness = factors.get("brightness", 0)
    contrast = factors.get("contrast", 0)
    augmentation = keras.Sequential([
        keras.layers.RandomBrightness(brightness, seed=cfg["seed"] + 1),
        keras.layers.RandomContrast(contrast, seed=cfg["seed"] + 2),
    ]) if brightness or contrast else None

    def prepare(batch, targets):
        batch = tf.cast(batch, tf.float32)
        if augmentation is not None:
            batch = augmentation(batch, training=True)
        return batch, targets

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(len(labels), seed=cfg["seed"])
        .batch(cfg["batch_size"])
        .map(prepare)
        .prefetch(1)
    )


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
    fine_tune = bool(cfg.get("fine_tune_from"))
    if fine_tune or cfg.get("resume_from"):
        model, prior_history = _resume_training(keras, cfg, split_path, checkpoint)
        if not fine_tune:
            best_loss = float(prior_history["val_loss"].min())
    else:
        model = _build_model(cfg)
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
    fit_inputs = {"x": x_train, "y": y_train, "batch_size": cfg["batch_size"]}
    if fine_tune:
        fit_inputs = {
            "x": _fine_tune_dataset(tf, keras, x_train, y_train, cfg),
            "validation_batch_size": cfg["batch_size"],
        }
    history = model.fit(
        validation_data=(x_val, y_val),
        initial_epoch=len(prior_history), epochs=cfg["epochs"],
        callbacks=callbacks, verbose=2, **fit_inputs,
    )
    full_history = pd.concat([prior_history, pd.DataFrame(history.history)], ignore_index=True)
    selection = full_history.iloc[len(prior_history):] if fine_tune else full_history
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
        "best_epoch": int(selection["val_loss"].idxmin() + 1),
        "resumed_at_epoch": len(prior_history),
        "test_evaluated": False,
    }
    (reports / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
