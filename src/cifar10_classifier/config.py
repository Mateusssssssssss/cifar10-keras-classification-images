"""Configuração do experimento."""
import re
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "seed", "validation_fraction", "batch_size", "epochs", "learning_rate",
        "patience", "reduce_lr_patience", "dropout", "weight_decay",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Parâmetros ausentes: {sorted(missing)}")
    for key in ("run_name", "resume_from"):
        name = config.get(key, "")
        if not isinstance(name, str) or (
            name and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
        ):
            raise ValueError(f"{key} deve conter apenas letras, números, hífens e underscores")
    resume_from = config.get("resume_from", "")
    run_name = config.get("run_name", "")
    if resume_from and (not run_name or resume_from.casefold() == run_name.casefold()):
        raise ValueError("Use um run_name próprio para a continuação")
    return config
