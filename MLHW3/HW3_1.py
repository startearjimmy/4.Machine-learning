# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:08:01 2018

@author: Jim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  linear_model
"""
read file
"""
csvfile = "Concrete_Data.csv"
data = pd.read_csv(csvfile)
(row,column)=data.shape
X_train=data['Age (day)'] #input
X_trainsecond=np.zeros([row,2])
for i in range(row):
    X_trainsecond[i][0]=X_train[i]
    X_trainsecond[i][1]=0
Y_train=data['Concrete compressive strength(MPa, megapascals) '] #output
#print(X_trainsecond.shape,Y_train.shape)

"""
lineaer regression
"""
regr = linear_model.LinearRegression()
regr.fit(X_trainsecond,Y_train)
print('weight: ',regr.coef_)
print('bias: ',regr.intercept_)
print('accuracy(r2_score): ',regr.score(X_trainsecond,Y_train))
"""
visualization, scatter plot and linear regression expresiion
"""
x=([0,0],[0,0])
x[0][0]=min(X_train)
x[1][0]=max(X_train)
y=([0,0])
y[0] = regr.coef_*x[0] + regr.intercept_
y[1] = regr.coef_*x[1] + regr.intercept_
plt.scatter(X_train,Y_train)
plt.plot(x,y)
plt.show()
