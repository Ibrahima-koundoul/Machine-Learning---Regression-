# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 00:28:37 2020

@author: ibrah
"""
"""Polynomial REGRESSION vs SVR"""

#We want to know the salary of an new employee in his previous company
#he was at level 6 and know he want to be at level 7

#1st Method Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the librairies
dataset = pd.read_csv("Position_Salaries.csv")
print(dataset.head())
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#in order to leverage the maximum data we don't need to split our dataset
#into train and test set
#No missing values
#no data scaling (we don't need it)

from sklearn.preprocessing import PolynomialFeatures

polynom = PolynomialFeatures(degree = 4)
X_poly = polynom.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly,y)

#Visualizing the results from the 1st Method PR
plt.figure()
plt.scatter(X,y,color='red')
plt.plot(X,regressor_poly.predict(polynom.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the results from the 1st Method PR (clear)
X_grid = np.arange(1,10,0.1)
X_grid = X_grid.reshape((X_grid.shape[0],1))
plt.figure()
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor_poly.predict(polynom.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#2nd Method SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
y=y.reshape((y.shape[0],1))
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
y = y.ravel()
print("{}".format(X))
print(y)

from sklearn.svm import SVR
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X,y)

#Visualizing the results from the 2nd Method SVR
plt.figure()
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor_svr.predict(X)), color='blue')
plt.title('Truth or Bluff (SVR METHOD)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
#Visualizing the results from the 2nd Method SVR more clear
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((X_grid.shape[0],1))
plt.figure()
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(sc_X.inverse_transform(X_grid),sc_y.inverse_transform(regressor_svr.predict(X_grid)), color='blue')
plt.title('Truth or Bluff (SVR METHOD MORE CLEAR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

Salary_predicted_by_poly = regressor_poly.predict(polynom.transform([[6.5]]))
Salary_predicted_by_SVR = regressor_svr.predict(sc_X.transform([[6.5]]))

comparaison = np.concatenate((sc_y.inverse_transform(Salary_predicted_by_SVR),Salary_predicted_by_poly),axis = 0)
print(comparaison)

