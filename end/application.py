import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

application = Flask(__name__)
app = application


import os

base_path = os.path.dirname(__file__)
scaler = pickle.load(open(os.path.join(base_path, 'notebooks/scaler.pkl'), 'rb'))
lin_reg_model = pickle.load(open(os.path.join(base_path, 'notebooks/lin_reg.pkl'), 'rb'))


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = lin_reg_model.predict(new_data_scaled)

        return render_template('home.html', results=(result[0]))
    else:
       return render_template('index.html') 

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)