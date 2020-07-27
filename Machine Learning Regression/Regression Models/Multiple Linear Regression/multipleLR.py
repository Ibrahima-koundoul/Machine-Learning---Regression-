# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:00:06 2020

@author: ibrah
"""


"""MULTIPlE LINEAR REGRESSION""" #first method : ALL IN

#importing librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset = pd.read_csv('50_Startups.csv')
#rearrange the dataset
dataset=dataset[['State','R&D Spend','Administration','Marketing Spend','Profit']]
X=dataset.iloc[:,0:-1].values
Y=dataset.iloc[:,-1].values



#taking care of missing data
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=0,strategy='mean')
imputer.fit(X[:,1:])
X[:,1:]=imputer.transform(X[:,1:])


#Encoding categorical data : dependant variable state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])],'passthrough')
X=np.array(ct.fit_transform(X))


#splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state =0)

#ALL-IN METHOD : LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred)
y_pred = y_pred.reshape((y_pred.shape[0],1))
Y_test = np.array(Y_test)
Y_test = Y_test.reshape((10,1))


result = np.concatenate((y_pred,Y_test),axis=1)
print(result)

#OPTIMAL METHOD
import statsmodels.api as sm
X = np.append(np.ones((50,1)).astype(int),X,axis=1)
X_opt = np.array(X[:,[0,1,2,3,4,5,6]],dtype=float)
regressor_OLS = sm.OLS(Y,X_opt).fit()
print(regressor_OLS.summary())

