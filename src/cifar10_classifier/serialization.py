"""Exportação do checkpoint final para inferência."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from .data import CLASS_NAMES

DEFAULT_MODEL_DIR = Path("models/final")
MODEL_FILE = "cifar10_residual.keras"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_export(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["classes"] != list(CLASS_NAMES):
        raise ValueError("A ordem das classes diverge do CIFAR-10")
    if _sha256(directory / MODEL_FILE) != manifest["model_sha256"]:
        raise ValueError("O SHA-256 do modelo exportado diverge do manifesto")
    return manifest


def export_model(directory: Path = DEFAULT_MODEL_DIR) -> Path:
    from tensorflow import keras

    frozen = yaml.safe_load(Path("configs/final_test.yaml").read_text(encoding="utf-8"))
    source = Path("models") / frozen["run_name"] / "best_model.keras"
    if _sha256(source) != frozen["checkpoint_sha256"]:
        raise ValueError("O checkpoint de origem diverge do manifesto final")
    if directory.exists():
        exported = verify_export(directory)
        if any(exported.get(key) != value for key, value in frozen.items()):
            raise ValueError("O diretório já contém uma exportação de outro checkpoint")
        return directory / MODEL_FILE

    model = keras.models.load_model(source, compile=False)
    images = np.random.default_rng(42).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    expected = model(images, training=False).numpy()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".export-") as temporary:
        staged = Path(temporary)
        artifact = staged / MODEL_FILE
        model.save(artifact)
        restored = keras.models.load_model(artifact, compile=False)
        np.testing.assert_allclose(restored(images, training=False).numpy(), expected, atol=1e-7)
        manifest = {
            **frozen,
            "classes": list(CLASS_NAMES),
            "model_sha256": _sha256(artifact),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staged.rename(directory)
    return directory / MODEL_FILE
