# tests/test_logic.py
import io
from PIL import Image
from lab1.logic.predictor import predict_class, resize_image, get_image_size


def create_test_image_bytes(color=(255, 0, 0), size=(64, 64)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_predict_class_returns_valid_class():
    b = create_test_image_bytes()
    cls = predict_class(b)
    assert isinstance(cls, str)
    assert cls


def test_resize_and_size():
    b = create_test_image_bytes(size=(100, 200))
    orig = get_image_size(b)
    assert orig == (100, 200)
    resized = resize_image(b, 32, 32)
    new_size = get_image_size(resized)
    assert new_size == (32, 32)
