# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:59:05 2022

@author: fanghonghong
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt



data = np.genfromtxt("C:/Users/fanghonghong/Documents/ml/data.csv", delimiter=',')

x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)

x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
# create model
model = LinearRegression()
model.fit(x_data,y_data)
#plot
plt.plot(x_data,y_data,'b')
plt.plot(x_data, model.predict(x_data),'r')
plt.show()