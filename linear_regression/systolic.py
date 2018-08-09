import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# plot the data to see how it looks like
plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

# method for calculating the r-squared
def get_r2(X, Y):
	w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
	Yhat = np.dot(X, w)

	d1 = Y - Yhat
	d2 = Y - Y.mean()
	r2 = 1 - d1.dot(d1) / d2.dot(d2)

	return r2


df['ones'] = 1
Y = df['X1']#.as_matrix()
X = df[['X2', 'X3', 'ones']]#.as_matrix()
X1 = df[['X2', 'ones']]#.as_matrix()
X2 = df[['X3', 'ones']]#.as_matrix()

print("r2 for X1 only:", get_r2(X1, Y))
print("r2 for X2 only:", get_r2(X2, Y))
print("r2 for X1 & X2:", get_r2(X,  Y))