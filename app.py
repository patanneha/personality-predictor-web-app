
import tensorflow as tf
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150
MODEL_PATH = 'my_personality_classifier.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
LABEL_COLUMNS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

app = Flask(__name__)
CORS(app)

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded.")

def preprocess_text(text):
    """Preprocesses a single string of text for the model."""
    text_list = [text]
    sequences = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

@app.route('/predict', methods=['POST'])
def predict():
    """Handle POST requests to the /predict endpoint."""
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No "text" field provided in request.'}), 400

        user_text = data['text']
        processed_text = preprocess_text(user_text)
        pred_probs = model.predict(processed_text)

        results = []
        for i, trait in enumerate(LABEL_COLUMNS):
            probability = float(pred_probs[0][i])
            class_name = "High" if probability > 0.5 else "Low"
            results.append({
                'trait': trait.capitalize(),
                'probability': probability,
                'class': class_name
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 for deployment, port is often set by the platform (like Render)
    app.run(host='0.0.0.0', port=5000, debug=False)