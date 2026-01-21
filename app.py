"""
Wine Cultivar Origin Prediction - Flask Web Application
========================================================
This Flask application loads the trained model and provides a web interface
for users to input wine chemical properties and get cultivar predictions.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
model_data = joblib.load(model_path)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# Cultivar names mapping
CULTIVAR_NAMES = {
    0: 'Cultivar 1',
    1: 'Cultivar 2',
    2: 'Cultivar 3'
}

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get input values from form
        input_values = []
        for feature in features:
            value = float(request.form.get(feature, 0))
            input_values.append(value)
        
        # Create numpy array and scale
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get cultivar name and confidence
        cultivar_name = CULTIVAR_NAMES[prediction]
        confidence = prediction_proba[prediction] * 100
        
        # Get all probabilities
        probabilities = {
            CULTIVAR_NAMES[i]: f"{prob*100:.1f}%"
            for i, prob in enumerate(prediction_proba)
        }
        
        return render_template(
            'index.html',
            features=features,
            prediction=cultivar_name,
            confidence=f"{confidence:.1f}%",
            probabilities=probabilities,
            input_values=dict(zip(features, input_values))
        )
    
    except Exception as e:
        return render_template(
            'index.html',
            features=features,
            error=f"Error making prediction: {str(e)}"
        )

if __name__ == '__main__':
    print("="*50)
    print("Wine Cultivar Origin Prediction System")
    print("="*50)
    print(f"Model loaded: {model_path}")
    print(f"Features: {features}")
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5001")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5001)
