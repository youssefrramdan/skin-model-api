from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import time
import os
from pydantic import BaseModel

app = FastAPI(
    title="Model Classification API",
    description="API for classification using TFLite model",
    version="1.0.0"
)

# Load TFLite model
MODEL_PATH = "model_quantized.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Replace these with your model's class names
CLASS_NAMES = ["class1", "class2"]  # Update with your actual class names

class ImageRequest(BaseModel):
    image_url: str

def preprocess_image(image: Image.Image):
    # Update these parameters according to your model's requirements
    img = image.convert("RGB").resize((224, 224))  # Adjust size as needed
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0  # Normalize
    return arr

@app.post("/predict")
async def predict(request: ImageRequest):
    total_start = time.time()

    try:
        # Step 1: Download image
        download_start = time.time()
        response = requests.get(request.image_url, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": f"Invalid content type: {content_type}"})
        img = Image.open(BytesIO(response.content))
        download_end = time.time()

        # Step 2: Predict
        pred_start = time.time()
        img_array = preprocess_image(img)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        pred_end = time.time()

        # Step 3: Return result
        total_end = time.time()

        return JSONResponse({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "timings": {
                "download": round(download_end - download_start, 3),
                "prediction": round(pred_end - pred_start, 3),
                "total": round(total_end - total_start, 3)
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/info")
async def info():
    return {
        "app_name": "Model Classification API",
        "model": "Classification using TensorFlow Lite",
        "classes": CLASS_NAMES,
        "input_shape": input_details[0]['shape'],
        "endpoints": {
            "/predict": "POST - Provide image URL to classify",
            "/info": "GET - Get information about this API"
        }
    }

# For running locally or on cloud platforms
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
