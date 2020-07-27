# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:21:43 2020

@author: ibrah
"""

#importing the librairies
#exceptionnaly we don't need to split our data into training and test set
#we got a small dataset
#we want to leverage the maximum data in order to train our model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset.head())
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#training the Linear regression in whole dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#training the polynomial linear regression in whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly,y)

#visualizing the 2 methods
plt.figure()
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Level')
plt.ylabel("Salary")
plt.show()


plt.figure()
plt.scatter(X,y,color="red")
plt.plot(X,regressor_poly.predict(X_poly),color="blue")
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Level')
plt.ylabel("Salary")
plt.show()

print(regressor_poly.predict(poly_reg.fit_transform([[6.5]])))
