import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(Y, T):
	return -np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))

def classification_rate(Y, T):
	return np.mean(np.round(Y) == T)

N = 4
D = 2

X = np.array([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
])

T = np.array([0, 1, 1, 0])

ones = np.ones((N, 1))
xy = np.array([X[:,0] * X[:,1]]).T
Xb = np.concatenate((ones, xy, X), axis = 1)

w = np.random.randn(D + 2)

errors = []
learning_rate = 0.01
l2 = 0.1
for i in range(5000):
	Y = sigmoid(Xb.dot(w))
	error = cross_entropy(Y, T)
	errors.append(error)
	if i % 100 == 0:
		print(error)
	w -= learning_rate * (Xb.T.dot(Y - T) + l2 * w)

print('Final w:', w)
print('Final classification rate:', classification_rate(Y, T))

plt.plot(errors)
plt.title('Cross-entropy Error')
plt.show()