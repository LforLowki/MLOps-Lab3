# lab1/models/onnx_classifier.py
from pathlib import Path
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

class ONNXClassifier:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        with open(self.labels_path, "r") as fh:
            self.labels = json.load(fh)

    def preprocess(self, pil_image: Image.Image, size: int = 224) -> np.ndarray:
        img = pil_image.convert("RGB").resize((size, size))
        arr = np.array(img).astype("float32") / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, pil_image: Image.Image) -> str:
        x = self.preprocess(pil_image)
        outputs = self.session.run(None, {self.input_name: x})
        logits = outputs[0]

        idx = int(np.argmax(logits, axis=1)[0])
        return self.labels[idx] if idx < len(self.labels) else str(idx)
