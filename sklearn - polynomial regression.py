# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:31:09 2022

@author: fanghonghong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("C:/Users/fanghonghong/Documents/ml/job.csv",delimiter=',')
x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)
plt.show()

x_data = data[1:,1,np.newaxis]
model = LinearRegression()
model.fit(x_data, y_data)

#plot
plt.plot(x_data,y_data,'b')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()

#Define polynomial regression, the value of degree can adjust the characteristics of the polynomial
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_data)
x_poly
#Define a regression model
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)


plt.plot(x_data, y_data, 'b')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()