# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:13:42 2020

@author: ibrah
"""


"""SIMPLE LINEAR REGRESSION"""

#importing librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0:1].values
Y=dataset.iloc[:,-1].values

#splitting data into train & test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2,random_state=0)

#training the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predict salary from test set
Y_pred = regressor.predict(X_test)

#visualizing LR with the train sets
plt.figure()
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs YOE (training sets)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


#visualizing LR with the test sets
plt.figure()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='blue')
plt.title('Salary vs YOE (test sets)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

#printing the score
print(regressor.score(X_test,Y_test))

"""predict employe'salary with 34yoe"""
print(regressor.predict([[34]]))

"""Get the equation"""
a=regressor.coef_
b=regressor.intercept_
print("Salary = ","{}".format(a),"YOE","+","{}".format(b))
