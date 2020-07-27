# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:46:36 2020

@author: ibrah
"""


"""DECISION TREE REGRESSION"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
dtr =DecisionTreeRegressor(random_state =0)
dtr.fit(X,y)
print(dtr.predict([[6.5]]))

#visualizing the decision tree regression (higher resolution)
plt.figure()
plt.subplot(2,1,1)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((X_grid.shape[0],1))
plt.scatter(X,y, color ='blue')
plt.plot(X,dtr.predict(X),color='red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Decison TREE REGRESSION )')
plt.subplot(2,1,2)
plt.scatter(X,y, color ='blue')
plt.plot(X_grid,dtr.predict(X_grid),color='red')
plt.xlabel('Level')
plt.ylabel('Salary')

plt.show()

