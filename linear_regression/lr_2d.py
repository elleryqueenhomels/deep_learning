import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load the data
# ref: df['x1x2'] = df.apply(lambda row: row['x1'] * row['x2'], axis = 1)
# pass in axis=1 so the apply function across over each row not each column
df = pd.read_csv('data_2d.csv', sep = ',', header = None)
df.columns = ['X1', 'X2', 'Y']
df['ones'] = 1
X = df[['X1', 'X2', 'ones']].as_matrix()
Y = df['Y'].as_matrix()

# load the data
# X = []
# Y = []
# for line in open('data_2d.csv'):
# 	x1, x2, y = line.split(',')
# 	X.append([float(x1), float(x2), 1])
# 	Y.append(float(y))
# X = np.array(X)
# Y = np.array(Y)

# It does not work for the command 'python', but 'ipython'
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# It does not work for the command 'python', but 'ipython'
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)
ax.plot(sorted(X[:, 0]), sorted(X[:, 1]), sorted(Yhat))
plt.show()

# calculate the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("r-squared:", r2)
print("w =", w)