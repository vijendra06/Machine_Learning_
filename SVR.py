import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[ :, 1:2].values
y= dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scx= StandardScaler()
scy= StandardScaler()
x= scx.fit_transform(x)
y= scy.fit_transform(y)

from sklearn.svm import SVR
clf= SVR( kernel= 'rbf')
clf.fit(x,y)

ypred= scy.inverse_transform(clf.predict(scx.transform(np.array([[6.5]]))))

plt.scatter(x,y, color='blue')
plt.plot( x, clf.predict(x), color= 'red')
plt.show()