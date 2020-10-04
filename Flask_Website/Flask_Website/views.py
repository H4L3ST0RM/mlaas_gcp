"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from Flask_Website import app
import os
from flask import Flask, jsonify, request
import json
#from prediction import predict
from flask import Flask
import pickle
import numpy as np


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )


@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


gender = pickle.load(open('./gender_predictor.pkl', 'rb'))
height = pickle.load(open('./height_predictor.pkl', 'rb'))
weight= pickle.load(open('./weight_predictor.pkl', 'rb'))


@app.route('/api')
def server_is_up():
    return 'server is up'

@app.route('/api/predict_weight', methods=['POST'])
def predict_weight():
    to_predict = request.json
    x = np.array([float(to_predict[col]) for col in ["gender","height"]])
    x = x.reshape(1,-1)
    print(to_predict)
    y_pred = weight.predict(x)[0]
    return jsonify({"predict weight":y_pred})

@app.route('/api/predict_height', methods=['POST'])
def predict_height():
    to_predict = request.json
    x = np.array([float(to_predict[col]) for col in ["gender","weight"]])
    x = x.reshape(1,-1)
    print(to_predict)
    y_pred = height.predict(x)[0]
    return jsonify({"predict height":y_pred})

@app.route('/api/predict_gender', methods=['POST'])
def predict_gender():
    to_predict = request.json
    x = np.array([float(to_predict[col]) for col in ["height","weight"]])
    x = x.reshape(1,-1)
    print(to_predict)
    y_pred = int(gender.predict(x)[0])
    return jsonify({"predict gender":y_pred})


@app.route('/api/gender_model', methods=['GET'])
def get_gender_model():
    inter = gender.intercept_.tolist()
    coefs = gender.coef_.tolist()
    params = {"model": str(gender),"intercept": inter, "coefficients":coefs}
    params.update(gender.get_params())
    return jsonify(params)


@app.route('/api/height_model', methods=['GET'])
def get_height_model():
    inter = height.intercept_.tolist()
    coefs = height.coef_.tolist()
    params = {"model": str(height),"intercept": inter, "coefficients":coefs}
    params.update(height.get_params())
    return jsonify(params)


@app.route('/api/weight_model', methods=['GET'])
def get_weight_model():
    inter = weight.intercept_.tolist()
    coefs = weight.coef_.tolist()
    params = {"model": str(weight),"intercept": inter, "coefficients":coefs}
    params.update(weight.get_params())
    return jsonify(params)