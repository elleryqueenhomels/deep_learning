import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(T, Y):
	#return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()
	N = T.shape[0]
	J = 0
	for i in range(N):
		if T[i] == 1:
			J -= np.log(Y[i])
		else:
			J -= np.log(1 - Y[i])
	return J

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

T = np.array([0] * 50 + [1] * 50)

ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis = 1)

w = np.random.randn(D + 1)

learning_rate = 0.1
l2 = 0.1

for i in range(100):
	Y = sigmoid(Xb.dot(w))

	if i % 10 == 0:
		print(cross_entropy(T, Y))

	w -= learning_rate * (Xb.T.dot(Y - T) + l2 * w)

print("Final w:", w)