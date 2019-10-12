# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 02:14:52 2019

@author: Asus
"""
import numpy as np

X=([1, 2],[3, 4])
print(X[0])
print(X[1])

theta1=
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