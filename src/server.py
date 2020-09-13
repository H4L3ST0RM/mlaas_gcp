import os
from flask import Flask, jsonify, request

import json
#from prediction import predict
import pickle
import numpy as np
HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def flask_app():
    app = Flask(__name__)
    gender = pickle.load(open('gender_predictor.pkl', 'rb'))
    height = pickle.load(open('height_predictor.pkl', 'rb'))
    weight= pickle.load(open('weight_predictor.pkl', 'rb'))


    @app.route('/', methods=['GET'])
    def server_is_up():
        return 'server is up'

    @app.route('/predict_weight', methods=['POST'])
    def predict_weight():
        to_predict = request.json
        x = np.array([float(to_predict[col]) for col in ["gender","height"]])
        x = x.reshape(1,-1)
        print(to_predict)
        y_pred = weight.predict(x)[0]
        return jsonify({"predict weight":y_pred})

    @app.route('/predict_height', methods=['POST'])
    def predict_height():
        to_predict = request.json
        x = np.array([float(to_predict[col]) for col in ["gender","weight"]])
        x = x.reshape(1,-1)
        print(to_predict)
        y_pred = height.predict(x)[0]
        return jsonify({"predict height":y_pred})

    @app.route('/predict_gender', methods=['POST'])
    def predict_gender():
        to_predict = request.json
        x = np.array([float(to_predict[col]) for col in ["height","weight"]])
        x = x.reshape(1,-1)
        print(to_predict)
        y_pred = gender.predict(x)[0]
        return jsonify({"predict gender":y_pred})



    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')