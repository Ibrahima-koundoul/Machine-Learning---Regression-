# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:45:15 2020

@author: ibrah
"""


"""Random forest Regression"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=500,random_state=0)
rfr.fit(X,y)
print(rfr.predict([[6.5]]))

#visualizing the decision tree regression (higher resolution)
plt.figure()
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((X_grid.shape[0],1))
plt.scatter(X,y, color ='blue')
plt.plot(X_grid,rfr.predict(X_grid),color='red')
plt.xlabel('Level')
plt.ylabel('Salary')

plt.show()