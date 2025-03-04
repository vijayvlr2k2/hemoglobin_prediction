import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the saved model
with open("hb_from_vitals_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Hemoglobin Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract SpO2 and Heart Rate from input
        spO2 = float(data["SpO2"])
        heart_rate = float(data["Heart Rate"])

        # Prepare data for prediction
        input_data = pd.DataFrame({"SpO2": [spO2], "Heart Rate": [heart_rate]})

        # Predict Hemoglobin
        predicted_hemoglobin = model.predict(input_data)[0]

        # Return response
        return jsonify({"Hemoglobin": predicted_hemoglobin})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
