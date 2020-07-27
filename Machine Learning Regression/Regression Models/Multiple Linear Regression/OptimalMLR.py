# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:37:28 2020

@author: ibrah
"""


"""OPTIMAL MULTIPLE LINEAR REGRESSION BACKWARD ELIMINATION"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
#rearrange my dataset
dataset = dataset[['State','R&D Spend','Administration','Marketing Spend','Profit']]

X = dataset.iloc[:,0:-1].values
y=dataset.iloc[:,-1].values

#taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(0,'mean')
imputer.fit(X[:,1:])
X[:,1:]=imputer.transform(X[:,1:])

#Encoding dependant variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])],'passthrough')
X=np.array(ct.fit_transform(X))
#print(X)

#splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state =0)

#Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)


"""OPTIMAL METHOD"""
import statsmodels.api as sm
X=np.append(np.ones((50,1)).astype(int),X,axis=1)
X_opt = np.array(X[:,[0,1,2,3,4,5,6]],dtype=float)
regressor_OLS = sm.OLS(endog =y, exog=X_opt).fit()
#print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,1,2,3,4,6]],dtype=float)
regressor_OLS = sm.OLS(endog =y, exog=X_opt).fit()
#print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,2,3,4,6]],dtype=float)
regressor_OLS = sm.OLS(endog =y, exog=X_opt).fit()
#print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,2,4,6]],dtype=float)
regressor_OLS = sm.OLS(endog =y, exog=X_opt).fit()
#print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,4,6]],dtype=float)
regressor_OLS = sm.OLS(endog =y, exog=X_opt).fit()
print(regressor_OLS.summary())

"""MODEL IS READY best features to predict profit are : R&D & Marketing""" 