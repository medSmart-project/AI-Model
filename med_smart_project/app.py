from flask import Flask, request, jsonify
from medicine_stock_prediction_model import model, scaler  # Import the model and scaler from model.py


# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Receive the request data (expecting JSON format)
    data = request.get_json(force=True)
    demand = data['demand']
    quantity = data['quantity']

    # Scale the input data
    input_data_scaled = scaler.transform([[demand, quantity]])

    # Predict using the trained model
    prediction = model.predict(input_data_scaled)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

# Run the Flask app
if __name__ == 'main':
    app.run(debug=True)
