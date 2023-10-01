from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os


# Generate Flask application to manage the server and prediction requests
app = Flask(__name__)

# Read model from parent folder
mycwd = os.getcwd()
os.chdir('..')
with open(os.getcwd() + '\\model.pkl', "rb") as file:
    model = pickle.load(file)
os.chdir(mycwd)


# Create function to make prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Request data acquisition (data needs to be shared in json object format)
    df = pd.json_normalize(data)  # Transform json object for input data to dataframe
    pred = model.predict(df)  # Calculate prediction using loaded model
    return jsonify(pred.tolist())  # Apply jsonify to prediction list to transform response to json object


# Enable server in port 8000
if __name__ == '__main__':
    app.run(port=8000)
