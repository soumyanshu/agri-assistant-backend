import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf # We still need this, but we use it lightly
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# --- 1. LOAD THE TFLITE MODEL (Lightweight) ---
print("⏳ Loading TFLite Model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite Model Loaded!")

# --- 2. CLASS NAMES ---
class_names = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy',
    'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot (Gray Leaf Spot)', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Haunglongbing (Citrus Greening)',
    'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper Bell - Bacterial Spot', 'Pepper Bell - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy',
    'Soybean - Healthy',
    'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 
    'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # TFLite expects float32
    image = image.astype(np.float32) 
    image = image / 255.0  
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        # TFLite usually expects 224x224, check your training if it fails
        processed_image = prepare_image(image, target_size=(224, 224))
        
        # --- PREDICTION USING TFLITE ---
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        prediction = output_data[0]
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        result = class_names[index]
        
        remedy = "Consult a local agriculture expert."
        if "Healthy" in result:
            remedy = "Your plant is healthy! Keep monitoring water and sunlight."
        elif "Bacterial" in result:
            remedy = "Use copper-based bactericides and avoid overhead watering."
        elif "Fungus" in result or "Blight" in result or "Rot" in result:
            remedy = "Apply fungicides like Mancozeb. Remove infected leaves immediately."
        elif "Virus" in result:
            remedy = "No chemical cure. Remove infected plants to prevent spread."

        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'remedy': remedy
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)