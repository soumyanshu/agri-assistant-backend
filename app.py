import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allows the website to talk to this server

# --- 1. LOAD YOUR MODEL ---
print("⏳ Loading AI Model... Please wait...")
# Ensure the filename matches exactly what you downloaded
model = tf.keras.models.load_model('agri_smart_full_model.h5')
print("✅ Model Loaded Successfully!")

# --- 2. DEFINE THE 38 CLASSES ---
# These match the PlantVillage dataset order exactly
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
    """Prepares the user's image for the AI model"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Read and process the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image, target_size=(224, 224))
        
        # Make Prediction
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        result = class_names[index]
        
        # Simple Remedy Logic (You can expand this!)
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
    # Run the server on port 5000
    app.run(debug=True, port=5000)