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

from sklearn.naive_bayes import GaussianNB
clf= GaussianNB()
clf.fit(xtrain, ytrain)

ypred= clf.predict(xtest)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest, ypred)

acc= clf.score(xtest, ytest)