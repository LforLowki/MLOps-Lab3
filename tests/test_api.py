# tests/test_api.py
import io
from fastapi.testclient import TestClient
from lab1.api.api import app
from PIL import Image

client = TestClient(app)


def make_image_bytes(size=(64, 64)):
    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_no_resize():
    img = make_image_bytes()
    files = {"file": ("test.jpg", img, "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    j = r.json()
    assert "predicted_class" in j
    assert "original_size" in j
    assert j["original_size"]["width"] == 64


def test_predict_with_resize():
    img = make_image_bytes(size=(100, 80))
    img.seek(0)
    files = {"file": ("test.jpg", img, "image/jpeg")}
    data = {"width": "32", "height": "32"}
    r = client.post("/predict", files=files, data=data)
    assert r.status_code == 200
    j = r.json()
    assert j["processed_size"]["width"] == 32
