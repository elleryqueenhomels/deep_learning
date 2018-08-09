import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def classification_rate(Y, T):
	return np.mean(Y == T)

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0] * (D - 3))

T = np.round(sigmoid(X.dot(true_w) + np.random.randn(N) * 0.5))

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 3.0
for i in range(50000):
	Y = sigmoid(X.dot(w))
	w -= learning_rate * (X.T.dot(Y - T) + l1 * np.sign(w))

	cost = -(T*np.log(Y) + (1-T)*np.log(1-Y)).mean() + l1*np.abs(w).mean()
	costs.append(cost)

plt.plot(costs, label = 'costs')
plt.legend()
plt.show()

print('Final w:', w)
print('classification rate:', classification_rate(np.round(Y), T))

plt.plot(true_w, label = 'true w')
plt.plot(w, label = 'w_map')
plt.legend()
plt.show()