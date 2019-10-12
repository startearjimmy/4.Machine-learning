# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:10:50 2018

@author: Jim
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import pandas as pd
csvfile = "Concrete_Data.csv"
data = pd.read_csv(csvfile)
(row,column)=data.shape
x1 = np.array(data['Cement (component 1)(kg in a m^3 mixture)'])
x2 = np.array(data['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
y = np.array(data['Concrete compressive strength(MPa, megapascals) '])
x=np.c_[x1,x2]
y=np.c_[y]
print('x:\n',x,'\ny:\n',y)

#set learnin_rate & iterations & matrixs of weights
learnin_rate=0.00000001
iterations=50000
weights=np.random.rand(3,1)
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
        #print('p:\n',prediction,'\nw:\n',weights)
        weights_history[i,:]=weights.T
        cost_history[i]=cal_cost(weights,x,y)

    return weights, cost_history,weights_history

#main function
xs=np.c_[np.ones(len(x)),x]
weights,cost_history,weights_history=gradient_descent(xs,y,weights,learnin_rate,iterations)
X1, X2 = np.meshgrid(x1, x2)
ys=weights[0]+(weights[1]*X1)+(weights[2]*X2)
print('final weights= \n',weights,'\n')
#plt.scatter(x[:,0],y)
#plt.scatter(x[:,1],y)
#plt.plot(x,ys)
#print(x[:,0].shape)
#print(x[:,1].shape)
#print(ys.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X1, X2, ys, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.scatter(x1, x2, y, c='r', marker='.')
plt.show()


