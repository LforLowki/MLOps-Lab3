# lab1/api/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path

from lab1.logic.predictor import predict_class, resize_image, get_image_size

app = FastAPI(title="MLOps Lab1 API")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    width: str | None = Form(None),
    height: str | None = Form(None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    original_width, original_height = get_image_size(content)

    # Convert to int if possible, else use original
    try:
        w = int(width) if width else original_width
        h = int(height) if height else original_height
    except ValueError:
        raise HTTPException(status_code=400, detail="Width and height must be integers")

    output_bytes = resize_image(content, w, h)
    predicted = predict_class(output_bytes)
    new_size = get_image_size(output_bytes)

    return JSONResponse(
        {
            "predicted_class": predicted,
            "original_size": {"width": original_width, "height": original_height},
            "processed_size": {"width": new_size[0], "height": new_size[1]},
        }
    )


@app.get("/health")
def health():
    return {"status": "ok"}
