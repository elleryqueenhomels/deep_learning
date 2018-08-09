import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('data_poly.csv', sep = ',', header = None)
df.columns = ['X', 'Y']
df['ones'] = 1
df['X^2'] = df.apply(lambda row: row['X'] * row['X'], axis = 1)
X = df[['ones', 'X', 'X^2']].as_matrix()
Y = df['Y'].as_matrix()

# load the data
# X = []
# Y = []
# for line in open('data_poly.csv'):
# 	x, y = line.split(',')
# 	x = float(x)
# 	X.append([1, x, x * x])
# 	Y.append(float(y))

# X = np.array(X) # covert to numpy
# Y = np.array(Y) # covert to numpy

# check the scatter
plt.scatter(X[:, 1], Y)
plt.show()

# calculate the weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot the data
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

# calculate the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared:", r2)