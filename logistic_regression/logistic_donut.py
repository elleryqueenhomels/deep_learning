# Logistic Regression Classifier for the donut problem

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(Y, T):
	return -np.sum(T*np.log(Y) + (1-T)*np.log(1-Y)) # or use np.mean

def classification_rate(Y, T):
	return np.mean(Y == T)

N = 1000
D = 2

R_inner = 5.0
R_outer = 10.0

# distance from center is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))
X_inner = np.concatenate(([R1 * np.cos(theta)], [R1 * np.sin(theta)])).T

R2 = np.random.randn(int(N/2)) + R_outer
theta = 2*np.pi*np.random.random(int(N/2))
X_outer = np.concatenate(([R2 * np.cos(theta)], [R2 * np.sin(theta)])).T

X = np.concatenate((X_inner, X_outer))
T = np.array([0]*(N/2) + [1]*(N/2))

plt.scatter(X[:, 0], X[:, 1], c = T)
plt.show()

# add a column of ones to X in order to add bias term
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))

# add a column of r = sqrt(x^2 + y^2)
r = np.zeros((N, 1))
for i in range(N):
	r[i] = np.sqrt(X[i,:].dot(X[i,:]))
Xb = np.concatenate((ones, r, X), axis = 1)

# randomly initialize the weights
W = np.random.randn(D + 2)

# use Gradient Descent to optimize the weights
learning_rate = 0.001
l2 = 0.1
costs = []
for i in range(5000):
	# calculate the Logistic Regression Model output
	Y = sigmoid(Xb.dot(W))
	cost = cross_entropy(Y, T)
	costs.append(cost)
	if i % 100 == 0:
		print(cost)

	# Gradient Descent weights update with L2 Regularization
	W -= learning_rate * (Xb.T.dot(Y - T) + l2 * W)

print('Fianl w:', W)
print('Final classification_rate:', classification_rate(np.round(Y), T))

plt.plot(costs)
plt.title('Cross-Entropy Error per iteration')
plt.show()