from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import numpy as np
from joblib import load
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('best_decision_tree_model.pkl', 'rb'))

# Load your machine learning model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        size = request.form['size']
        fuel = request.form['fuel']
        distance = request.form['distance']
        decibel = request.form['decibel']
        airflow = request.form['airflow']
        frequency = request.form['frequency']

        total = np.array([[size, fuel, distance, decibel, airflow, frequency]])
        y_test = model.predict(total)

        if y_test[0] == 0:
            result = "The fire is in the extension state"
        else:
            result = "none"

        return render_template('home.html', result=result)

    except Exception as e:
        return render_template('home.html', result="invalid input")

@app.route("/result", methods=['GET'])  # Handle GET requests for /result
def show_result():
    return render_template('home.html', result="")

if __name__ == "__main__":
    app.run(debug=True)

