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
    for key in ("run_name", "resume_from", "fine_tune_from"):
        name = config.get(key, "")
        if not isinstance(name, str) or (
            name and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
        ):
            raise ValueError(f"{key} deve conter apenas letras, números, hífens e underscores")
    resume_from = config.get("resume_from", "")
    fine_tune_from = config.get("fine_tune_from", "")
    if resume_from and fine_tune_from:
        raise ValueError("Escolha apenas um entre resume_from e fine_tune_from")
    parent = resume_from or fine_tune_from
    run_name = config.get("run_name", "")
    if parent and (not run_name or parent.casefold() == run_name.casefold()):
        raise ValueError("Use um run_name próprio para a continuação")
    if "photometric_augmentation" in config:
        factors = config["photometric_augmentation"]
        if not fine_tune_from or not isinstance(factors, dict) or set(factors) != {"brightness", "contrast"}:
            raise ValueError("Defina brightness e contrast em photometric_augmentation no ajuste fino")
        if any(type(value) not in (float, int) or not 0 <= value <= 1 for value in factors.values()):
            raise ValueError("Os fatores de brilho e contraste devem estar entre 0 e 1")
    if "bicubic_size" in config:
        size = config["bicubic_size"]
        if not fine_tune_from or type(size) is not int or size <= 32:
            raise ValueError("bicubic_size deve ser um inteiro maior que 32 no ajuste fino")
    return config
