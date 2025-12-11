from pathlib import Path
from PIL import Image
from lab1.models.onnx_classifier import ONNXClassifier
import io

# BASE_MODELS_DIR = Path("/app/models")
BASE_MODELS_DIR = (
    Path("/app/models")
    if Path("/app/models").exists()
    else Path(__file__).parent.parent.parent / "models"
)
print(str(BASE_MODELS_DIR))
MODEL_PATH = BASE_MODELS_DIR / "best_model.onnx"
LABELS_PATH = BASE_MODELS_DIR / "labels.json"

classifier = ONNXClassifier(
    model_path=str(MODEL_PATH),
    labels_path=str(LABELS_PATH),
)


def resize_image(
    pil_image: bytes | Image.Image, width: int = 224, height: int | None = None
) -> bytes:
    if isinstance(pil_image, bytes):
        pil_image = Image.open(io.BytesIO(pil_image))
    if height is None:
        height = width
    img = pil_image.resize((width, height))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def get_image_size(pil_image: bytes | Image.Image):
    if isinstance(pil_image, bytes):
        pil_image = Image.open(io.BytesIO(pil_image))
    return pil_image.size  # (width, height)


def predict_class(pil_image: bytes | Image.Image) -> str:
    if isinstance(pil_image, bytes):
        pil_image = Image.open(io.BytesIO(pil_image))
    return classifier.predict(pil_image)
