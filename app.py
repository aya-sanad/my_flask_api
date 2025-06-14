from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('C://flask//ai//model_new.h5')
with open('C://flask//ai//tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 100

def get_confidence_label(pred):
    """Assign confidence label based on predicted probability."""
    if pred >= 0.9:
        return "Definitely Haram"
    elif pred >= 0.7:
        return "Likely Haram"
    elif pred >= 0.5:
        return "Probably Haram"
    elif pred >= 0.3:
        return "Probably Halal"
    elif pred >= 0.1:
        return "Likely Halal"
    else:
        return "Definitely Halal"

@app.route('/', methods=['GET'])
def index():
    return (
        "API is running. Use POST /predict with JSON body: "
        '{ "text": "ingredient1, ingredient2" }'
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Check content type
    if not request.content_type or 'application/json' not in request.content_type:
        return jsonify({"error": "Content-Type must be application/json"}), 415


    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON data: {e}"}), 400

    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON body"}), 400

    input_text = data['text']
    if not input_text.strip():
        return jsonify({"error": "'text' field cannot be empty"}), 400

    # Split ingredients by comma, strip whitespace
    ingredients = [ing.strip() for ing in input_text.split(",") if ing.strip()]

    if not ingredients:
        return jsonify({"error": "No valid ingredients found in 'text'"}), 400

    results = {}
    classifications = []

    for ingredient in ingredients:
        # Convert ingredient text to padded sequence
        sequence = tokenizer.texts_to_sequences([ingredient])
        padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

        # Predict probability (assuming binary classification, sigmoid output)
        prediction = float(model.predict(np.expand_dims(padded[0], axis=0))[0][0])
        label = get_confidence_label(prediction)

        results[ingredient] = {
            "label": label,
            "probability": round(prediction, 4)
        }

        classifications.append(label)

    # Overall classification logic
    overall_classification = (
        "Haram" if any("Haram" in label for label in classifications) else "Halal"
    )
    results["overall_classification"] = overall_classification

    return jsonify(results)

if __name__ == "__main__":
    print("Flask API is running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
