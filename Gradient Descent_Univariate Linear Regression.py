# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 18:59:26 2022

@author: fanghonghong
"""

import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("C:/Users/fanghonghong/Documents/ml/data.csv", delimiter=',')
# observe the shape of data
data.shape
#observe the distribute of data by plot
print('Correlation coefficient:',np.corrcoef(data[:,0], data[:,1]))
plt.scatter(data[:,0], data[:,1])
plt.show()
#Solving Linear Equations in One Variable Using Gradient Descent
#1)set initial variables
theta0 = 0
theta1 = 0
learning_rate = 0.0001
epoch = 30
m = data.shape[0]
#2)Cyclic adjustment of theta0 and theta1
x_data = data[:,0]
y_data = data[:,1]
for i in range(epoch):
    theta0_grad = np.sum(theta0+theta1*x_data-y_data)/m
    theta1_grad = np.sum((theta0+theta1*x_data-y_data)*x_data)/m
    theta0 = theta0 - learning_rate * theta0_grad
    theta1 = theta1 - learning_rate * theta1_grad
    
print(f'theta0={theta0},theta1={theta1}')

plt.scatter(x_data, y_data)
min_value = np.min(x_data) - 5
max_value = np.max(x_data) + 5
x_line = np.array([min_value, max_value])
y_line = theta0 + theta1 * x_line
plt.plot(x_line, y_line, c='red')