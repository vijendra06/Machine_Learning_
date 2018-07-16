#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:59:48 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[ :, [2,3]].values
y= dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split
xtrain, xtest,ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
xtrain= scx.fit_transform(xtrain)
xtest=  scx.transform(xtest)

#ytest= scy.fit_transform(y)

from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier()
clf.fit(xtrain, ytrain)

ypred=clf.predict(xtest)

acc= clf.score(xtest, ytest)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest, ypred)
