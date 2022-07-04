# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 21:36:50 2022

@author: fanghonghong
"""

import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

data = genfromtxt("C:/Users/fanghonghong/Documents/ml/longley.csv",delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1]

# create model
# generate 50 values
alphas_to_test = np.linspace(0.001,1)
# Create the model and save the error values
model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
model.fit(x_data, y_data)

print(model.coef_)
print(model.intercept_)

print(model.alpha_)
#error values
print(model.cv_values_.shape)

#plot
#The relationship between the ridge coefficient and the error
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
#The location of the selected ridge coefficient value
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)),'ro')
plt.show()

model.predict(x_data[1].reshape(1,-1))