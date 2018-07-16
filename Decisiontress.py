#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 13:46:48 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[ :, 1:2].values
y= dataset.iloc[:, 2:3].values

'''from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
scy= StandardScaler()
x= scx.fit_transform(x)
y= scy.fit_transform(y)'''

from sklearn.tree import DecisionTreeRegressor
clf= DecisionTreeRegressor( random_state= 0)
clf.fit( x, y)

ypred=  clf.predict(6.5)

#decision tree is non continuous model
xgrid= np.arange(min(x), max(x), 0.01)
xgrid= xgrid.reshape(len(xgrid), 1)
plt.scatter(x,y, color='red')
plt.plot( xgrid, clf.predict(xgrid), color='blue')
plt.show()
