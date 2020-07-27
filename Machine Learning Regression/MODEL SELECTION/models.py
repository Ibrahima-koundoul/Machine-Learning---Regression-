# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:13:20 2020

@author: ibrah
"""


"""HOW TO EVALUATE our regression MODELS and mostly to select the best one"""
"""We're trying to predict the energy output PE with different regression models"""
#we don't have categorical data
#neither missing values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

#no features scaling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""1st model - MULTIPLE LINEAR REGRESSION (ALL-in Method / Backward Elimination)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape((1914,1))
y_test = y_test.reshape((y_test.shape[0],1))
results = np.concatenate((y_pred,y_test),axis=1)
print(results)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #score : 0.9325

#2nd Method : Backward Elimination
import statsmodels.api as asm
X = np.append(np.ones((X.shape[0],1)),X,axis=1)
X_opt = np.array(X[:,[0,1,2,3,4]],dtype=float)
regressor_opt = asm.OLS(y,X_opt).fit()
#print(regressor_opt.summary())
#ALL p-value are < to 0,05"""

"""2nd Model : Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
X_poly = PolynomialFeatures(degree=2)
X = X_poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape((1914,1))
y_test = y_test.reshape((y_test.shape[0],1))
results = np.concatenate((y_pred,y_test),axis=1)
print(results)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #score : 0.9421
"""



"""3rd MODEL SVM Model we need to apply features scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()

y_train = y_train.reshape((y_train.shape[0],1))
X_train=sc_X.fit_transform(X_train)
y_train=sc_y.fit_transform(y_train)
y_train = y_train.ravel()

y_test = y_test.reshape((y_test.shape[0],1))
X_test = sc_X.transform(X_test)
y_test = sc_y.transform(y_test)
y_test = y_test.ravel()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #score: 0.9325
"""


""" 4th Model the Decision Tree model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #score : 0.9223
"""

"""Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #score : 0.9653
"""
"""The best model is our RandomForestModel with a score of 0.9653"""