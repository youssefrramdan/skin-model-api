from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO
import time
import os

app = Flask(__name__)

# Load both TFLite models
STATUS_MODEL_PATH = "status_model.tflite"
CANCER_MODEL_PATH = "Skin.tflite"

if not os.path.exists(STATUS_MODEL_PATH) or not os.path.exists(CANCER_MODEL_PATH):
    raise FileNotFoundError("Required model files not found")

# Initialize TFLite interpreters
status_interpreter = tflite.Interpreter(model_path=STATUS_MODEL_PATH)
cancer_interpreter = tflite.Interpreter(model_path=CANCER_MODEL_PATH)

status_interpreter.allocate_tensors()
cancer_interpreter.allocate_tensors()

# Get input and output tensors for both models
status_input_details = status_interpreter.get_input_details()
status_output_details = status_interpreter.get_output_details()

cancer_input_details = cancer_interpreter.get_input_details()
cancer_output_details = cancer_interpreter.get_output_details()

# Define class names
STATUS_CLASSES = ['Normal', 'Affected']
CANCER_CLASSES = ['MEL', 'BCC', 'SCC', 'AK', 'BKL', 'DF', 'VASC']

def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def get_prediction(interpreter, input_details, output_details, img_array):
    """Get prediction from specified model"""
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image URL from request
        request_data = request.get_json()
        if not request_data or 'image_url' not in request_data:
            return jsonify({"error": "No image URL provided"}), 400

        image_url = request_data['image_url']

        # Download and process image
        response = requests.get(image_url, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return jsonify({"error": f"Invalid content type: {content_type}"}), 400

        img = Image.open(BytesIO(response.content))
        img_array = preprocess_image(img)

        # Get status prediction first
        status_predictions = get_prediction(status_interpreter, status_input_details, status_output_details, img_array)
        status_class = STATUS_CLASSES[np.argmax(status_predictions[0])]
        status_confidence = float(np.max(status_predictions[0]))

        result = {
            "status": status_class,
            "status_confidence": round(status_confidence, 4)
        }

        # If affected, get cancer type prediction
        if status_class == 'Affected':
            cancer_predictions = get_prediction(cancer_interpreter, cancer_input_details, cancer_output_details, img_array)
            cancer_type = CANCER_CLASSES[np.argmax(cancer_predictions[0])]
            cancer_confidence = float(np.max(cancer_predictions[0]))

            result.update({
                "cancer_type": cancer_type,
                "cancer_confidence": round(cancer_confidence, 4)
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app_name": "Skin Cancer Detection API",
        "models": {
            "status_model": "Detects if skin is Normal or Affected",
            "cancer_model": "Classifies type of skin cancer if Affected"
        },
        "status_classes": STATUS_CLASSES,
        "cancer_classes": CANCER_CLASSES,
        "endpoints": {
            "/predict": "POST - Provide image URL to classify",
            "/info": "GET - Get information about this API",
            "/ping": "GET - Health check endpoint"
        }
    })

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "status_model": True,
            "cancer_model": True
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
