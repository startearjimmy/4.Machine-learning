import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    m = X.shape[0] #5000
    a1 = np.insert(X, 0, 1, axis=1)

    w1=a1.T
    z2 = theta1.dot(w1)
    z2=z2.T

    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    
    w2=a2.T
    z3 = theta2.dot(w2)
    z3=z3.T

    h  = sigmoid(z3)

    return a1, z2, a2, z3, h
    
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))
    
    return J
    
# initial setup
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
#print(X.shape,y.shape)
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
#print(theta1.shape,theta2.shape)
#print(len(theta1[:,1:]),len(theta2[:,1:]))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
#print(a1.shape, z2.shape, a2.shape, z3.shape, h.shape)

#print(cost(params, input_size, hidden_size, num_labels, X, y, learning_rate))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    #Write codes here
    J=cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    grad=[]
    for k in range(10):
        for j in range(26):
            #grad.append((y[:,k]-h[:,k])*(sigmoid_gradient(z3[:,k]))*a2[:,j])
            unknown=(y[:,k]-h[:,k])*(sigmoid_gradient(z3[:,k]))*a2[:,j]
            print(unknown.shape)
    for k in range(25):
        for j in range(401):
            for i in range(10):
                unknown=theta2[i,k]*(y[:,i]-h[:,i])*(sigmoid_gradient(z3[:,i]))*(sigmoid_gradient(z2))*a1[:,j]
                print(unknown.shape)
                #grad.append(theta2[i,k]*(y[:,i]-h[:,i])*(sigmoid_gradient(z3[:,i]))*(sigmoid_gradient(z2))*a1[:,j])
    return J , grad 

from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
