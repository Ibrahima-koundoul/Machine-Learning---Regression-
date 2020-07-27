# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:58:52 2020

@author: ibrah
"""


#SuperVectorRegression : NONLINEAR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
print(Y)
print(X)

#we don't need to split our dataset into train and test set because 
#we want to leverage the maximum data

#we need to apply features scaling to avoid the independant variable to be neglected
#by the SVR model
#but before we'll apply some tranformation for ou dependant variable vector
#Because the StandardScalar class expect and 2D array
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
Y = Y.reshape((Y.shape[0],1))

X = sc_X.fit_transform(X)
Y =sc_Y.fit_transform(Y)

print(Y)
print(X)

from sklearn.svm import SVR
regressor=SVR('rbf')
Y=Y.ravel()
regressor.fit(X,Y)

print(sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))
