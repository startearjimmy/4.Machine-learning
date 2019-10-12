# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:54:26 2019

@author: Asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=(hidden_size * (input_size + 1) + num_labels * (hidden_size + 1))) - 0.5) * 0.2
print(params.shape)
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
print(X.shape[0])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
print(theta1.shape)
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
print(theta2.shape)

a1 = np.append(1,X[0])
print(a1.shape)
z2 = theta1.dot(a1)
print(z2.shape)
a2 = np.append(1,sigmoid(z2))
print(a2.shape)
z3 = theta2.dot(a2)
print(z3.shape)
h  = sigmoid(z3)
print(h.shape)