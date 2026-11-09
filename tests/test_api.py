import io
from importlib import import_module

import numpy as np
import pytest
from cifar10_classifier.data import CLASS_NAMES

TestClient = pytest.importorskip("fastapi.testclient").TestClient
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("python_multipart")
api = import_module("cifar10_classifier.api")


@pytest.fixture
def client(monkeypatch):
    def predict(batch, training):
        assert training is False
        assert batch.shape == (1, 32, 32, 3)
        assert batch.dtype == np.uint8
        np.testing.assert_array_equal(batch, 127)
        return np.eye(len(CLASS_NAMES))[[3]]

    manifest = {"classes": list(CLASS_NAMES), "run_name": "residual_60_cpu", "best_epoch": 34}
    monkeypatch.setattr(api, "_load_predictor", lambda _: (predict, manifest))
    with TestClient(api.create_app()) as session:
        yield session


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model": "residual_60_cpu", "checkpoint_epoch": 34,
    }


def test_predict_resizes_and_converts_grayscale_image(client):
    image = io.BytesIO()
    Image.new("L", (48, 64), 127).save(image, format="PNG")
    response = client.post("/predict", files={"file": ("input.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["class_id"] == 3
    assert result["class_name"] == "cat"
    assert result["confidence"] == result["probabilities"]["cat"] == 1.0
    assert list(result["probabilities"]) == list(CLASS_NAMES)


def test_predict_rejects_invalid_upload(client):
    assert client.post("/predict").status_code == 422
    response = client.post("/predict", files={"file": ("invalid.png", b"abc", "image/png")})
    assert response.status_code == 415


def test_predict_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 2)
    response = client.post("/predict", files={"file": ("large.png", b"abc", "image/png")})
    assert response.status_code == 413


def test_predict_rejects_excessive_dimensions(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 15)
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    response = client.post("/predict", files={"file": ("large.png", image.getvalue(), "image/png")})
    assert response.status_code == 413
