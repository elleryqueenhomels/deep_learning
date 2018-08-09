import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
data = pd.read_csv('data_1d.csv', sep = ',', header = None)
X = data[:][0].as_matrix()
Y = data[:][1].as_matrix()

xSum = X.sum()
xMean = X.mean()
yMean = Y.mean()
a = (X.dot(Y) - yMean * xSum) / (X.dot(X) - xMean * xSum)
b = yMean - a * xMean

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculate r-squared
d1 = Y - Yhat
d2 = Y - yMean
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print('the r-squared is: %s' % r2)