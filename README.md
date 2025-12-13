[![CICD](https://github.com/LforLowki/MLOps-Lab3/actions/workflows/CICD.yml/badge.svg)](https://github.com/LforLowki/MLOps-Lab3/actions/workflows/CICD.yml)
---

# MLOps Lab 3 – Experiment Tracking, Model Versioning & Deployment

This project corresponds to **Lab 3** of the MLOps course.  
It extends **Lab 1** and **Lab 2** by replacing the random predictor with a **real deep learning image classifier**, tracking experiments with **MLflow**, serializing the selected model to **ONNX**, and deploying the system using **Docker, Render, and Hugging Face Spaces**.

---

## Project Overview

The goal of this lab is to **train, track, version, select, and deploy** a deep learning model following MLOps best practices.

The system consists of:

- **Training pipeline** using transfer learning (MobileNetV2)
- **Experiment tracking & model registry** with MLflow
- **Model selection & serialization** to ONNX
- **Inference API** built with FastAPI + ONNX Runtime
- **Continuous Integration & Deployment (CI/CD)** with GitHub Actions
- **Backend deployment** on Render (Docker-based)
- **Frontend GUI** using Gradio hosted on Hugging Face Spaces

---

## Model & Data

- **Dataset**: Oxford-IIIT Pet Dataset  
- **Model**: MobileNetV2 (pretrained on ImageNet)
- **Input size**: 224 × 224 RGB images
- **Training strategy**:
  - Freeze feature extractor
  - Replace final classifier layer
  - Cross-entropy loss
  - Adam optimizer
- **Hardware**:
  - Training: CPU / GPU (locally or CI)
  - Inference: CPU-only (Render & Hugging Face)

---

## Repository Structure

```

.
├── lab1/
│   ├── api/              # FastAPI backend
│   ├── cli/              # CLI utilities
│   ├── logic/            # Prediction logic (ONNX inference)
│   ├── models/           # Training, export, ONNX runtime code
│   └── preprocessing/    # Image preprocessing utilities
├── models/               # Exported ONNX model & labels (runtime)
├── data/                 # Dataset (ignored by git)
├── tests/                # Unit tests
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
└── .github/workflows/    # CI/CD pipeline

```

---

## Training & Experiment Tracking

Training is performed using **transfer learning** and tracked with **MLflow**.

### Training Script
- `lab1/models/train.py`
- Logs:
  - Hyperparameters
  - Training & validation metrics (per epoch)
  - Class labels (JSON)
- Registers models under a single name:
```

lab3_pet_model

````

### Run Training
```bash
make run-train
````

### MLflow UI

```bash
mlflow ui
```

---

## Model Selection & ONNX Export

The best model is selected automatically based on **validation accuracy**.

### Export Script

* `lab1/models/select_export.py`
* Responsibilities:

  * Query MLflow model registry
  * Select best model version
  * Load PyTorch model
  * Convert to CPU
  * Export to ONNX
  * Download class labels

### Output

```
models/
├── best_model.onnx
├── best_model.onnx.data
└── labels.json
```

---

## Inference API

The API uses **ONNX Runtime** for fast CPU inference.

### Backend

* **Framework**: FastAPI
* **Inference**: ONNX Runtime (CPUExecutionProvider)
* **Endpoint**:

  * `POST /predict`
  * `GET /health`

### Run API Locally

```bash
uv run uvicorn lab1.api.api:app --reload
```

---

## Docker & Deployment

### Docker Image

* Multi-stage build
* Runtime does **not** include PyTorch
* Only ONNX Runtime + FastAPI dependencies

### Deployment

* **Docker Hub**: stores built image
* **Render**: hosts the API
* Automatic redeployment via GitHub Actions

---

## Hugging Face Space (Frontend)

A **Gradio** app provides a web interface to the deployed API.

* Hosted on **Hugging Face Spaces**
* Uses the **Render API URL**
* No model files stored in HF (API-based inference)

### Example `app.py`

```python
import gradio as gr
import requests

API_URL = "https://<your-render-api>.onrender.com"

def predict_image(file):
    with open(file.name, "rb") as f:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.jpg", f, "image/jpeg")},
            timeout=20,
        )
    return response.json()

gr.Interface(
    fn=predict_image,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(),
    title="Oxford-IIIT Pet Classifier",
    description="Upload an image and get a prediction from the deployed API."
).launch()
```

---

## CI/CD Pipeline

The GitHub Actions pipeline performs:

1. Install dependencies
2. Format & lint code
3. Run tests
4. Train model
5. Export ONNX model
6. Build & push Docker image
7. Deploy API to Render
8. Push Gradio app to Hugging Face Space

---

## Git Ignore

The following directories are ignored:

```
data/
mlruns/
models/
plots/
results/
```

---

## Key Technologies

* Python 3.11
* PyTorch
* MLflow
* ONNX / ONNX Runtime
* FastAPI
* Docker
* GitHub Actions
* Render
* Gradio
* Hugging Face Spaces

---

## Author

Developed as part of the **MLOps course – Lab 3**
Author: **LforLowki**
---
