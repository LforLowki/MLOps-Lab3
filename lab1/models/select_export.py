# lab1/models/select_export.py
"""
Query MLflow registered model versions, pick best by validation accuracy,
load the PyTorch model, convert to CPU, set eval, export to ONNX, download labels artifact.
"""

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import torch
import mlflow.pytorch

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)


def select_best_model(registered_name="lab3_pet_model"):
    client = MlflowClient(tracking_uri="file://" + str(Path("mlruns").resolve()))
    # get all versions for this model
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        raise RuntimeError("No registered model versions found.")

    # For each version, get run metrics and look for 'val_acc' last value
    best = None
    best_acc = -1.0
    for v in versions:
        run_id = v.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics
        val_acc = metrics.get("val_acc", None)
        # If val_acc is not logged per epoch but as single, it's okay.
        if val_acc is None:
            # try find any val_acc_* keys (fallback)
            val_acc = 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best = v

    print(f"Selected version {best.version} (run {best.run_id}) with val_acc={best_acc}")
    return best


def export_to_onnx(best_version, model_uri=None, output_path=OUTPUT_DIR / "best_model.onnx", opset=18):
    if model_uri is None:
        model_uri = f"runs:/{best_version.run_id}/model"
    # load the PyTorch model (mlflow.pytorch.load_model returns a torch.nn.Module)
    model = mlflow.pytorch.load_model(model_uri)
    model.to("cpu")
    model.eval()

    # Create dummy input (batch 1)
    import torch
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(model, dummy, str(output_path), opset_version=opset, input_names=["input"], output_names=["output"])
    print(f"Exported ONNX to {output_path}")
    return output_path


def download_labels(best_version, filename="labels.json", dest=OUTPUT_DIR / "labels.json"):
    client = MlflowClient(tracking_uri="file://" + str(Path("mlruns").resolve()))
    run_id = best_version.run_id
    local_path = client.download_artifacts(run_id, "labels/labels.json", dst_path=str(OUTPUT_DIR))
    # above returns path to file inside OUTPUT_DIR (or a folder) - ensure we have it at dest
    if isinstance(local_path, list):
        lp = Path(local_path[0])
    else:
        lp = Path(local_path)
    if lp.exists():
        # move/rename
        lp.rename(dest)
    print(f"Downloaded labels to {dest}")
    return dest


if __name__ == "__main__":
    best = select_best_model("lab3_pet_model")
    export_to_onnx(best)
    download_labels(best)
    print("Done.")
