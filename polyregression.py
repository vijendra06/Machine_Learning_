#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:06:48 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[ :, 1:2].values
y= dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures 
polyreg= PolynomialFeatures ( degree=4)
xpoly= polyreg.fit_transform(x)
#polyreg.fit(xpoly,y)
linreg2= LinearRegression()
linreg2.fit(xpoly,y)


#Linear Regression Plot
plt.scatter(x,y, color='blue')
plt.plot( x, lin_reg.predict(x), color= 'red')
plt.show()



#Polynomial Regression Plot
plt.scatter(x,y, color='blue')
plt.plot( x, linreg2.predict(polyreg.fit_transform(x)), color='red')
plt.show()
