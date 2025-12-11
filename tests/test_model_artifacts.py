# tests/test_model_artifacts.py
from pathlib import Path


def test_onnx_and_labels_exist():
    assert Path("models/best_model.onnx").exists(), "models/best_model.onnx missing"
    assert Path("models/labels.json").exists(), "models/labels.json missing"
