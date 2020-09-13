# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:36:59 2020

@author: johnc
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

df = pd.read_csv('C:/Users/johnc/Downloads/datasets_weight_height.csv')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

y = np.array(df['Gender'])
X = np.array(df[['Height', 'Weight']])
gender_predictor = LogisticRegression(random_state=0).fit(X, y)
#gender_predictor.predict(np.array([71,183]).reshape(1,-1))
pickle.dump(gender_predictor, open('gender_predictor.pkl','wb'))

y = np.array(df['Weight'])
X = np.array(df[['Gender', 'Height']])
weight_predictor = LinearRegression().fit(X, y)
pickle.dump(weight_predictor, open('weight_predictor.pkl','wb'))

#weight_predictor.predict(np.array([1,71]).reshape(1,-1))
y = np.array(df['Height'])
X = np.array(df[['Gender', 'Weight']])
height_predictor = LinearRegression().fit(X, y)
pickle.dump(height_predictor, open('height_predictor.pkl','wb'))

#height_predictor.predict(np.array([1,183]).reshape(1,-1))