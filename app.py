from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained Keras model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Load scaler for input preprocessing
SCALER_PATH = "scaler.save"
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

@app.route("/")
def home():
    """Render the main page with the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive input from the form, preprocess, predict cancer type, and render result.
    """
    try:
        # Get form data and convert to float
        features = [
            float(request.form["radius"]),
            float(request.form["texture"]),
            float(request.form["perimeter"]),
            float(request.form["area"]),
            float(request.form["smoothness"])
        ]

        # Prepare input for model
        input_features = np.array(features).reshape(1, -1)

        # Scale input if scaler exists
        if scaler:
            input_features = scaler.transform(input_features)

        # Predict probability of malignancy
        prediction = model.predict(input_features)
        malignant_prob = float(prediction[0][0])

        # Format result
        if malignant_prob > 0.5:
            result = f"Malignant (Cancerous) (Confidence: {malignant_prob:.2%})"
        else:
            result = f"Benign (Non-cancerous) (Confidence: {(1-malignant_prob):.2%})"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)