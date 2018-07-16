#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:08:48 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('50_Startups.csv')
x= dataset.iloc[ :, :-1].values
y= dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le= LabelEncoder()
x[:, 3]= le.fit_transform(x[:, 3])
ohe= OneHotEncoder( categorical_features=[3])
x= ohe.fit_transform(x).toarray()

x=x[:, 1:]

from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
clf= LinearRegression()
clf.fit(xtrain, ytrain)

ypred= clf.predict(xtest)


#backward Elimination
import statsmodels.formula.api as sm
x= np.append( arr= np.ones((50,1)).astype(int), values= x , axis=1)
xopt=x[:, [0,1,2,3,4,5]] 
regressor= sm.OLS(endog= y, exog=xopt).fit()
regressor.summary()

xopt=x[:, [0,3,4,5]] 
regressor= sm.OLS(endog= y, exog=xopt).fit()
regressor.summary()

xopt=x[:, [0,3,5]] 
regressor= sm.OLS(endog= y, exog=xopt).fit()
regressor.summary()