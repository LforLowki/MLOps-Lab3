# lab1/api/api.py
from fastapi import FastAPI, File, UploadFile, Form
from lab1.logic.predictor import predict_class, get_image_size, resize_image

app = FastAPI()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    width: int | None = Form(None),
    height: int | None = Form(None),
):
    content = await file.read()
    original_width, original_height = get_image_size(content)
    pred_class = predict_class(content)

    response = {
        "predicted_class": pred_class,
        "original_size": {"width": original_width, "height": original_height},
    }

    if width is not None and height is not None:
        # Pass width and height as separate arguments
        resized_content = resize_image(content, width, height)
        response["processed_size"] = {"width": width, "height": height}

    return response


@app.get("/health")
async def health():
    return {"status": "ok"}
