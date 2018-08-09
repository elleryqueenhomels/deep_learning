import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# center the first 50 points at (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0] * 50 + [1] * 50)

# add a column of ones for bias term
# ones = np.array([[1] * N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis = 1)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

w = np.array([0, 4, 4])

z = Xb.dot(w)
Y = sigmoid(z)

plt.scatter(np.array([1, 2, 3, 4, 5, 6]), np.array([6, 5, 4, 3, 2, 1]))
plt.show()

plt.scatter(X[:, 0], X[:, 1], c = T, s = 100, alpha = 0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

print(Y)
plt.scatter(Y, T, c = T, s = 100, alpha = 0.5)
plt.show()