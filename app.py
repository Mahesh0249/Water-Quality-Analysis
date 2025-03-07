from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
print("📌 Current Directory:", os.getcwd())

# Initialize Flask app
app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("water_quality_model.pkl"
"", "rb"))  # Make sure model.pkl exists in the same folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request contains a file (CSV Upload)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Read CSV file
            df = pd.read_csv(file)

            # Ensure the CSV file has the correct feature columns for the model
            predictions = model.predict(df)

            return jsonify({"predictions": predictions.tolist()})

        # If JSON input
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Convert JSON features into NumPy array
        features = np.array(data['features']).reshape(1, -1)  # Reshape for model input
        prediction = model.predict(features)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
