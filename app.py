from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from prometheus_client import generate_latest, Counter, Histogram

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("predict_request_count", "Total number of requests to the predict endpoint")
PREDICTION_TIME = Histogram("predict_latency_seconds", "Time taken to make predictions")

# Load trained model
model = load_model("models/best_lstm_model.h5")

@app.route('/predict', methods=['POST'])
@PREDICTION_TIME.time()  # Track latency
def predict():
    REQUEST_COUNT.inc()  # Increment request count
    data = request.json

    try:
        # Validate input
        features = np.array(data['features'])
        if features.shape != (1, 5, 5):
            return jsonify({"error": f"Invalid input shape: {features.shape}. Expected (1, 5, 5)"}), 400
        
        # Predict
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Add the /metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
