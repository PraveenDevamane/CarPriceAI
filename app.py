from flask import Flask, request, jsonify,send_from_directory
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + feature columns
model = pickle.load(open('car_price_model.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    data_df = pd.DataFrame([data])

    # Apply same encoding
    data_df = pd.get_dummies(
        data_df,
        columns=['fueltype', 'carbody', 'enginelocation', 'drivewheel',
                 'aspiration', 'enginetype', 'doornumber'],
        drop_first=True
    )
    data_df = data_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(data_df)[0]

    return jsonify({'predicted_price': round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(debug=True)






    from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare features array based on your model's requirements
    features = [
        data['symboling'], data['horsepower'], data['enginesize'],
        data['compressionratio'],
        # Add encoded categorical features
    ]
    
    prediction = model.predict([features])[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
