# to run the app type in the terminal - python app.py (Make sure you are in the directory)

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

# instantiation
app = Flask(__name__)
Swagger(app)

# load model file
classifier = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def welcome():
    return 'Welcome All................'

@app.route('/predict', methods=['GET'])
def heart_disease():
    
    """
    Heart Disease Prediction....
    This is using docstrings for specification
    
    ---
    parameters:
    
        -   name: male 
            in: query
            type: number
            required: true
        -   name: age
            in: query
            type: number
            required: true
        -   name: education
            in: query
            type: number
            required: true
        -   name: currentSmoker
            in: query
            type: number
            required: true
        -   name: cigsPerDay
            in: query
            type: number
            required: true
        -   name: BPMeds
            in: query
            type: number
            required: true
        -   name: prevalentStroke
            in: query
            type: number
            required: true
        -   name: prevalentHyp
            in: query
            type: number
            required: true
        -   name: diabetes
            in: query
            type: number
            required: true
        -   name: totChol
            in: query
            type: number
            required: true
        -   name: sysBP
            in: query
            type: number
            required: true
        -   name: diaBP
            in: query
            type: number
            required: true
        -   name: BMI
            in: query
            type: number
            required: true
        -   name: heartRate
            in: query
            type: number
            required: true
        -   name: glucose
            in: query
            type: number
            required: true
    
    responses:
        200:
            description: The output values
    
    """
    male = request.args.get('male')
    age = request.args.get('age')
    education = request.args.get('education')
    currentSmoker = request.args.get('currentSmoker')
    cigsPerDay = request.args.get('cigsPerDay')
    BPMeds = request.args.get('BPMeds')
    prevalentStroke = request.args.get('prevalentStroke')
    prevalentHyp = request.args.get('prevalentHyp')
    diabetes = request.args.get('diabetes')
    totChol = request.args.get('totChol')
    sysBP = request.args.get('sysBP')
    diaBP = request.args.get('diaBP')
    BMI	 = request.args.get('BMI')
    heartRate = request.args.get('heartRate')
    glucose = request.args.get('glucose')
    
    prediction = classifier.predict([[male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, 
                                    prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    
    return "The predicted value(s) are\t" + str(prediction)

# run
if __name__ == '__main__':
    app.run(debug=True)