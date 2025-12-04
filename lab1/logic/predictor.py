# src/logic/predictor.py
from io import BytesIO
from typing import Tuple
import random
from PIL import Image

CLASS_NAMES = ["cat", "dog", "bird", "car", "flower"]


def predict_class(image_bytes: bytes) -> str:
    """Dummy predictor: returns a random class from CLASS_NAMES."""
    return random.choice(CLASS_NAMES)


def resize_image(
    image_bytes: bytes, width: int, height: int, format: str = "JPEG"
) -> bytes:
    """Resize image to exact width x height and return bytes."""
    with BytesIO(image_bytes) as in_buf:
        with Image.open(in_buf) as img:
            img = img.convert("RGB")
            resized = img.resize((width, height))
            out_buf = BytesIO()
            resized.save(out_buf, format=format)
            out_buf.seek(0)
            return out_buf.read()


def get_image_size(image_bytes: bytes) -> Tuple[int, int]:
    with BytesIO(image_bytes) as b:
        with Image.open(b) as img:
            return img.size
