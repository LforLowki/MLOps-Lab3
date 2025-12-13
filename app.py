import gradio as gr
import requests

API_URL = "https://lab3-api-latest.onrender.com"

def predict_image(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": ("image.jpg", f, "image/jpeg")}
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=20
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(),
    title="Oxford-IIIT Pet Classifier",
    description="Upload an image and get a prediction from the deployed API."
)

if __name__ == "__main__":
    iface.launch()
