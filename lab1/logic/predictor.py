import numpy as np
from PIL import Image
import io


# Dummy classifier
class ONNXClassifier:
    def predict(self, image):
        x = self.preprocess(image)
        return "class_A"  # dummy


    def preprocess(self, image):
        # Convert bytes -> PIL.Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB").resize((224, 224))
        return np.array(image).astype(np.float32) / 255.0


classifier = ONNXClassifier()


def predict_class(image_bytes):
    return classifier.predict(image_bytes)


def resize_image(image_bytes, width, height):
    if isinstance(image_bytes, bytes):
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = image_bytes
    resized = image.resize((width, height))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG")
    return buf.getvalue()


def get_image_size(image_bytes):
    if isinstance(image_bytes, bytes):
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = image_bytes
    return image.size
