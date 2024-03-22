from flask import Flask, render_template, request
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)

# Set the static folder path
app.config['STATIC_FOLDER'] = 'static'

with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Define the prediction route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
   
# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Create a DataFrame from the JSON data
    df = pd.DataFrame({
        'TYPE': [data.get('property_type')],
        'SUBLOCALITY': [data.get('neighborhood')],
        'BEDS': [int(data.get('bedrooms'))],
        'BATH': [int(data.get('baths'))],
        'PROPERTYSQFT': [int(data.get('property_sqft'))]
    })

    # Make prediction
    predicted_log_price = xgb_model.predict(df)

    # Un-log the predicted price
    predicted_price = np.round(np.exp(predicted_log_price)[0], 2)

    # Return the predicted price as JSON response
    return jsonify({"prediction_text": "<b>House price should be ${:.2f}</b>".format(predicted_price)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)