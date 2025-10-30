import os
import joblib
from flask import Flask, request, jsonify
from preprocessing import preprocess_text

print("Loading model and vectorizer...")

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, "..", "models", "sentiment_model_lr.joblib")
    vectorizer_path = os.path.join(base_dir, "..", "models", "tfidf_vectorizer.joblib")

    baseline_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found.")
    print(f"Looked for model at: {model_path}")
    print(f"Looked for vectorizer at: {vectorizer_path}")
    baseline_model = None
    tfidf_vectorizer = None
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    baseline_model = None
    tfidf_vectorizer = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    if not baseline_model or not tfidf_vectorizer:
        return jsonify({"error": "Model is not loaded. Please check server logs."}), 500

    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": "Invalid JSON format."}), 400

    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' key in request JSON."}), 400
        
    raw_text = data['text']
    
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    try:
        processed_text = preprocess_text(raw_text)
        
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        
        prediction = baseline_model.predict(vectorized_text)
        
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        
        return jsonify({
            "sentiment": sentiment,
            "input_text": raw_text
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)