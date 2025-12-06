import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path


class ONNXClassifier:
    def __init__(self):
        model_path = Path("models/best_model.onnx")
        labels_path = Path("models/labels.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB").resize((224, 224))
        arr = np.array(image).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))   # CHW
        arr = np.expand_dims(arr, axis=0)    # batch dimension
        return arr

    def predict(self, image: Image.Image):
        x = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: x})
        logits = outputs[0][0]
        idx = int(np.argmax(logits))
        return self.labels[str(idx)]


# ------------------------------------------------------------
# The tests expect these FREE functions at module level
# ------------------------------------------------------------

def resize_image(image, size=(224, 224)):
    return image.resize(size)


def get_image_size(image):
    return image.size


def predict_class(image):
    classifier = ONNXClassifier()
    return classifier.predict(image)

