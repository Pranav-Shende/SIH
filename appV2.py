import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

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

# ---------------------------------------------------------------------
# Step 2: Load your machine learning model here when the app starts.
# Replace 'your_model_name.h5' with your actual model filename.
# This prevents the model from being reloaded for every request.
# ---------------------------------------------------------------------
try:
    # Check if the model file exists
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
    This function processes the image and uses the loaded model for prediction.
    """
    try:
        # Use Pillow to open the image from the in-memory data
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # -----------------------------------------------------------------
        # Step 3: Add your model's pre-processing logic here.
        # This is where you prepare the image for the model.
        # -----------------------------------------------------------------
        # The model was trained on 224x224 images with pixel values normalized to [0, 1].
        image = image.resize((224, 224))
        image_array = np.array(image)
        # Normalize the image array to a float32 type and scale to the range [0, 1]
        image_array = image_array.astype('float32') / 255.0
        # Add a batch dimension to match the model's input shape
        image_array = np.expand_dims(image_array, axis=0) 

        # -----------------------------------------------------------------
        # Step 4: Add your model's prediction logic here.
        # This is where you call your model to get the prediction.
        # -----------------------------------------------------------------
        if model:
            # FIX: Dynamically generate class names from the dictionary keys
            # This ensures consistency between the prediction output and the info dictionary.
            # The list is sorted alphabetically to match the behavior of
            # tf.keras.utils.image_dataset_from_directory
            class_names = sorted(list(CATTLE_BREEDS.keys()))
            
            predictions = model.predict(image_array)
            predicted_index = np.argmax(predictions)
            # The model's output index is now correctly mapped to the breed name
            predicted_breed = class_names[predicted_index]
        else:
            # Placeholder prediction logic if the model failed to load
            if image.width > image.height:
                predicted_breed = "Angus"
            else:
                predicted_breed = "Holstein"

        return predicted_breed

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
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