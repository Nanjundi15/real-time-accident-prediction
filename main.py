from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
model_paths = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model_fixed.pkl",
    "LSTM": "lstm_model.h5"
}

scaler = joblib.load("new_scaler_logistic.pkl")

# Load ML models
models = {name: joblib.load(path) for name, path in model_paths.items() if name != "LSTM"}
lstm_model = load_model(model_paths["LSTM"])

@app.route("/")
def home():
    return "Real-Time Traffic Accident Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON input
        df = pd.DataFrame([data])  # Convert to DataFrame

        # Scale input
        X_scaled = scaler.transform(df)

        predictions = {}

        # Predictions from traditional models
        for name, model in models.items():
            predictions[name] = model.predict(X_scaled).tolist()

        # LSTM Prediction
        X_lstm = np.expand_dims(X_scaled, axis=1)  # Reshape for LSTM
        lstm_pred = np.argmax(lstm_model.predict(X_lstm), axis=1)
        predictions["LSTM"] = lstm_pred.tolist()

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
