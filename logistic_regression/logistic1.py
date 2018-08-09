import numpy as np

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

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

Y = sigmoid(z)

# calculate the Cross-Entropy Error
def cross_entropy(T, Y):
	return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

print(cross_entropy(T, Y))

# try it with our closed-form solution
w = np.array([0, 4, 4])

# calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)

#c calculate the cross-entropy error
print(cross_entropy(T, Y))