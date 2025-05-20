from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO
import time
import os

app = Flask(__name__)

# Load optimized TFLite model (quantized for better performance and smaller size)
MODEL_PATH = "Skin_model_optimized.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names for skin cancer types
CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    total_start = time.time()

    try:
        # Get image URL from request
        request_data = request.get_json()
        if not request_data or 'image_url' not in request_data:
            return jsonify({"error": "No image URL provided"}), 400

        image_url = request_data['image_url']

        # Step 1: Download image
        download_start = time.time()
        response = requests.get(image_url, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return jsonify({"error": f"Invalid content type: {content_type}"}), 400
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

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app_name": "Skin Cancer Classification API",
        "model": "Skin Cancer Classification using TensorFlow Lite",
        "classes": CLASS_NAMES,
        "endpoints": {
            "/predict": "POST - Provide image URL to classify",
            "/info": "GET - Get information about this API",
            "/ping": "GET - Health check endpoint"
        }
    })

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "model": "skin"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
