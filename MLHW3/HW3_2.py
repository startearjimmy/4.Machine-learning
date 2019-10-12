# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:29:54 2018

@author: Jim
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
csvfile = "Concrete_Data.csv"
data = pd.read_csv(csvfile)
(row,column)=data.shape
x = np.array(data['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
y = np.array(data['Concrete compressive strength(MPa, megapascals) '])
x=np.c_[x]
y=np.c_[y]

#set learnin_rate & iterations & matrixs of weights
learnin_rate=0.00001
iterations=10000000
weights=np.random.rand(2,1)
print('parameter in the begining\n','learnin_rate= ',learnin_rate,'\n','iterations= ',iterations,'\n','weights= \n',weights,'\n')

# generate cost function
def cal_cost(weights,x,y):
    m=len(y)
    predictions = x.dot(weights)
    cost=(m/2)*np.sum(np.square(predictions-y))
    return cost

#generate Gradient Descend function
def gradient_descent(x,y,weights,learnin_rate,iterations):
    m=len(y)
    cost_history=np.zeros(iterations)
    weights_history=np.zeros((iterations,len(weights)))
    
    for i in range(iterations):
        prediction = np.dot(x,weights)
        weights=weights-(1/m)*learnin_rate*(x.T.dot(prediction-y))
        weights_history[i,:]=weights.T
        cost_history[i]=cal_cost(weights,x,y)
#        if (1/m)*learnin_rate*(x.T.dot(prediction-y))>weights[1][0]: 
#           learnin_rate/=10

    return weights, cost_history,weights_history

#main function
xs=np.c_[np.ones(len(x)),x]
weights,cost_history,weights_history=gradient_descent(xs,y,weights,learnin_rate,iterations)
ys=(weights[0]*xs[:,0])+(weights[1]*xs[:,1])
print('final weights= ','bias',weights[0],'weight',weights[1],'\n')
print('MSE= ',mean_squared_error(y, ys),'\n')
print('coefficient of determination= \n',LinearRegression().fit(x, y).score(x, y),'\n')
plt.scatter(x,y)
plt.plot(x,ys)
plt.show()