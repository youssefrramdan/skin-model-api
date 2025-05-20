import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the original model
model = load_model('Skin.keras')

# Define the optimization parameters
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Enable experimental features for further size reduction
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Representative dataset generator
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 256, 256, 3) * 255
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
with open('Skin_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

# Print size comparison
import os
original_size = os.path.getsize('Skin.keras') / (1024 * 1024)
new_size = os.path.getsize('Skin_model_optimized.tflite') / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Optimized model size: {new_size:.2f} MB")
print(f"Size reduction: {((original_size - new_size) / original_size * 100):.2f}%")

# Function to generate test data
def generate_test_data(num_samples=100):
    return np.random.rand(num_samples, 256, 256, 3) * 255

# Evaluate original model
test_data = generate_test_data()
original_predictions = model.predict(test_data)

# Load and evaluate TFLite model
interpreter = tf.lite.Interpreter(model_path='Skin_model_optimized.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate TFLite model
tflite_predictions = []
for i in range(len(test_data)):
    interpreter.set_tensor(input_details[0]['index'], test_data[i:i+1].astype(np.float32))
    interpreter.invoke()
    tflite_predictions.append(interpreter.get_tensor(output_details[0]['index']))
tflite_predictions = np.vstack(tflite_predictions)

# Calculate prediction differences
mean_diff = np.mean(np.abs(original_predictions - tflite_predictions))
max_diff = np.max(np.abs(original_predictions - tflite_predictions))

print("\nModel Accuracy Comparison:")
print(f"Mean absolute difference in predictions: {mean_diff:.4f}")
print(f"Maximum absolute difference in predictions: {max_diff:.4f}")

# Calculate agreement percentage
agreement = np.mean(np.argmax(original_predictions, axis=1) == np.argmax(tflite_predictions, axis=1)) * 100
print(f"Prediction agreement percentage: {agreement:.2f}%")
