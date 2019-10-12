# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:22:50 2019

@author: Asus
"""
from scipy.optimize import minimize, rosen, rosen_der

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
print(res.x[0])