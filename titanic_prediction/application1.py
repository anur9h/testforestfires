import pickle
from flask import Flask, request, render_template
import os

application1 = Flask(__name__)
app = application1

base_path = os.path.dirname(__file__)
scaler = pickle.load(open(os.path.join(base_path, 'models/scaler.pkl'), 'rb'))
log_reg_model = pickle.load(open(os.path.join(base_path, 'models/model.pkl'), 'rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Pclass = int(request.form.get('Pclass'))
        Sex = int(request.form.get('Sex'))
        Age = float(request.form.get('Age'))
        SibSp = int(request.form.get('SibSp'))
        Parch = int(request.form.get('Parch'))

        new_data_scaled = scaler.transform([[Pclass, Sex, Age, SibSp, Parch]])
        prediction = log_reg_model.predict(new_data_scaled)
        prediction_value = int(prediction[0])

        result = "Passenger does not survive" if prediction_value == 0 else "Passenger survives"
        return render_template('home.html', prediction=result)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
