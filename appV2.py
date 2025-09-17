import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:4028", "https://sih-6-yve3.onrender.com"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Dictionary with breed information
CATTLE_BREEDS = {
    "Banni buffalo": {
        "milk_yield": "Moderate",
        "traits": ["Buffalo", "Known for milk production in arid regions"]
    },
    "Brahman cattle": {
        "milk_yield": "Low",
        "traits": ["Heat-tolerant", "Humped back", "Loose dewlap", "Beef cow"]
    },
    "Brahmapuri buffalo": {
        "milk_yield": "Moderate",
        "traits": ["Buffalo", "Known for milk and draught purposes"]
    },
    "Dangi cattle": {
        "milk_yield": "Low",
        "traits": ["Draught animal", "Found in hilly regions", "Disease resistant"]
    },
    "Gir cattle": {
        "milk_yield": "High",
        "traits": ["Dairy cow", "Known for high milk yield", "Distinctive domed forehead"]
    },
    "Holstein Friesian cattle": {
        "milk_yield": "High",
        "traits": ["Dairy cow", "Distinctive black and white or red and white markings"]
    },
    "Jaffarabadi buffalo": {
        "milk_yield": "High",
        "traits": ["Buffalo", "Large and heavy breed", "Good milk yield"]
    },
    "Jersey cattle": {
        "milk_yield": "High (high butterfat content)",
        "traits": ["Dairy cow", "Small size", "Fawn-colored coat", "Gentle disposition"]
    },
    "Kankrej cattle": {
        "milk_yield": "Low to Moderate",
        "traits": ["Draught and milk breed", "Known for speed and power"]
    },
    "Kherigarh cattle": {
        "milk_yield": "Low",
        "traits": ["Draught animal", "Small size", "Found in northern India"]
    },
    "Kundi buffalo": {
        "milk_yield": "Moderate",
        "traits": ["Buffalo", "High-fat milk", "Found in Sindh region"]
    },
    "Mehsana buffalo": {
        "milk_yield": "High",
        "traits": ["Buffalo", "Cross of Murrah and Surti breeds"]
    },
    "Murrah buffalo": {
        "milk_yield": "High",
        "traits": ["Buffalo", "Known for high milk yield", "Jet black color"]
    },
    "Nagpuri buffalo": {
        "milk_yield": "Moderate",
        "traits": ["Buffalo", "Known for milk and draught purposes"]
    },
    "Nili-Ravi buffalo": {
        "milk_yield": "High",
        "traits": ["Buffalo", "Known for high milk yield", "Blue eyes"]
    },
    "Panchgavya cattle": {
        "milk_yield": "N/A",
        "traits": ["Religious and cultural significance", "Used for producing Panchgavya"]
    },
    "Red Sindhi cattle": {
        "milk_yield": "High",
        "traits": ["Dairy cow", "Known for high milk yield", "Red color"]
    },
    "Sahiwal": {
        "milk_yield": "High",
        "traits": ["Dairy cow", "Known for high milk yield", "Heat-tolerant"]
    },
    "Surti buffalo": {
        "milk_yield": "Moderate",
        "traits": ["Buffalo", "Known for high-fat milk", "Sickle-shaped horns"]
    },
    "Tharparkar": {
        "milk_yield": "Moderate",
        "traits": ["Dual-purpose breed", "Known for milk and draught purposes"]
    }
}

# Load ML model if available
try:
    model_path = os.path.join(os.getcwd(), 'cattle_buffalo_model.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Machine learning model loaded successfully.")
    else:
        model = None
        print("Warning: Model file not found. Using placeholder prediction.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}. Using placeholder prediction.")


def process_image_and_predict(image_data):
    """
    Processes the image and returns a predicted breed name.
    """
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if model:
            class_names = sorted(list(CATTLE_BREEDS.keys()))
            predictions = model.predict(image_array)
            predicted_index = np.argmax(predictions)
            predicted_breed = class_names[predicted_index]
        else:
            # Placeholder prediction if model not found
            predicted_breed = "Holstein" if image.width <= image.height else "Angus"

        return predicted_breed

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/')
def home():
    return "The server is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_data = file.read()
    predicted_breed = process_image_and_predict(image_data)

    if predicted_breed and predicted_breed in CATTLE_BREEDS:
        return jsonify({
            "prediction": predicted_breed,
            "info": CATTLE_BREEDS[predicted_breed]
        })
    else:
        return jsonify({"error": "Prediction failed or breed not found"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
