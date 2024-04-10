import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__) #Creating the Flask app
reg_model = pickle.load(open('regmodel.pkl', 'rb')) #Loading the model
scaler=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html') #Rendering the home.html template

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = reg_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(port=8000, debug=True) #Running the app
    #Run on port 8000 since Mac now runs its control center on port 5000 and 7000 which throws a 403 forbidden error with postman



