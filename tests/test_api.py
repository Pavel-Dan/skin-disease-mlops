import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _image_bytes() -> bytes:
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload


def test_predict_valid_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
    )
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment.")
    assert response.status_code == 200
    payload = response.json()
    assert "predicted_class" in payload
    assert "top3_predictions" in payload
    assert "disclaimer" in payload


def test_predict_invalid_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_empty_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
