#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:06:48 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Salary_Data.csv')
x= dataset.iloc[ :, :-1]
y= dataset.iloc[:, 1]

from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
clf= LinearRegression()
clf.fit(xtrain, ytrain)

ypred= clf.predict(xtest)
#train set 
plt.scatter( xtrain, ytrain, color='red')
plt.plot( xtrain,clf.predict(xtrain), color='blue' )
plt.show()
#test set
plt.scatter( xtest, ytest, color='red')
plt.plot( xtrain,clf.predict(xtrain), color='blue' )
plt.show()