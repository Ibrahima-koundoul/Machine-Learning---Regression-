# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:00:44 2020

@author: ibrah
"""


#DATA PREPROCESSING

"""Importing the librairies"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""

dataset = pd.read_csv('Data.csv')
#features
#generate a ndarray
X = dataset.iloc[:,:-1].values
#Dependant variable
#generate a vector
Y = dataset.iloc[:,-1].values
print(X)
print(Y)
"""deal with missing values"""
#1-option delete the rows with missing data (best way if mv are about 1%)
#2-replace by the average if the column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values =np.nan,strategy = 'mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

print(X)


"""Encoding categorical variable"""
#Encoding independant variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
##The fit_transform method doesn't return a ndarray
## we need to convert it on nparray

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#Encoding dependant variable
from sklearn.preprocessing import LabelEncoder
#y is a vector we don't need to transform it on nparray
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)


"""Splitting the dataset into test set and train set"""
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 1)
print(X_train)
print(Y_train)

print(X_test)
print(Y_test)
"""Features scaling"""
"put all features on the same scale"
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train)
print(X_test)