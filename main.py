from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
from io import BytesIO
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Skin_model_optimized.tflite")

logger.info(f"Looking for model in directory: {os.path.dirname(__file__)}")
logger.info(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Initialize TFLite interpreter
try:
    logger.info("Initializing model interpreter...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    img = image.convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0  # Normalize to [0,1]
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check if the file is an image
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return jsonify({"error": "File type not allowed. Please upload an image file"}), 400

        # Open and process the image
        img = Image.open(file.stream)
        img_array = preprocess_image(img)

        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get top prediction
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]

        result = {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "top_3_predictions": top_3_predictions
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app_name": "Skin Cancer Detection API",
        "model": "Skin Cancer Classification using TensorFlow Lite",
        "classes": CLASS_NAMES,
        "input_shape": "256x256 RGB image",
        "endpoints": {
            "/predict": "POST - Upload an image file for classification",
            "/info": "GET - Get information about this API",
            "/ping": "GET - Health check endpoint"
        }
    })

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
