import numpy as np


class ANN:
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type

	def initialize(self, D, K):
		self.W = []
		self.b = []

		L = len(self.layers)
		for i in range(L):
			if i == 0:
				w = np.random.randn(D, self.layers[i])
			else:
				w = np.random.randn(self.layers[i-1], self.layers[i])
			b = np.random.randn(self.layers[i])
			self.W.append(w)
			self.b.append(b)
		w = np.random.randn(self.layers[L-1], K)
		b = np.random.randn(K)
		self.W.append(w)
		self.b.append(b)

	def fit(self, X, Y, layers=None, learning_rate=10e-7, epochs=20000):
		if layers != None:
			self.layers = layers
		assert(self.layers != None)

		N, D = X.shape
		K = Y.shape[1]

		self.initialize(D, K)

		for i in range(epochs):
			Z = self.forward(X)
			self.backpropagation(Z, Y, learning_rate)
			if i % 100 == 0:
				print('%d:' % i, 'score =', self.score(np.argmax(Y, axis=1), np.argmax(Z[-1], axis=1)))

	def backpropagation(self, Z, T, learning_rate):
		L = len(self.W)
		delta = Z[-1] - T
		for i in range(L):
			self.W[L-1-i] -= learning_rate * Z[L-1-i].T.dot(delta)
			self.b[L-1-i] -= learning_rate * delta.sum(axis=0)
			if self.activation_type == 1:
				delta = delta.dot(self.W[L-1-i].T) * (Z[L-1-i] * (1 - Z[L-1-i]))
			else:
				delta = delta.dot(self.W[L-1-i].T) * ((1 + Z[L-1-i]) * (1 - Z[L-1-i]))

	def activation(self, a):
		if self.activation_type == 1:
			return 1 / (1 + np.exp(-a))
		else:
			return np.tanh(a)

	def softmax(self, a):
		expA = np.exp(a)
		return expA / expA.sum(axis=1, keepdims=True)

	def forward(self, X):
		Z = [X]
		for i in range(len(self.W)-1):
			Z.append(self.activation(Z[i].dot(self.W[i]) + self.b[i]))
		Z.append(self.softmax(Z[-1].dot(self.W[-1]) + self.b[-1]))
		return Z

	def predict(self, X):
		return np.argmax(self.forward(X)[-1], axis=1)

	def score(self, Y, P):
		return np.mean(Y == P)


def experiment():
	# create the data
	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes

	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2, 2])
	X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Y)
	# turn Y into an indicator matrix for training
	T = np.zeros((N, K))
	for i in range(N):
		T[i, Y[i]] = 1

	# let's see what it looks like
	# plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	# plt.show()

	model = ANN([3], activation_type=2)
	model.fit(X, T)
	print('score:', model.score(Y, model.predict(X)))


if __name__ == '__main__':
	from process import get_data, y2indicator
	X, Y = get_data()
	model = ANN([50, 50], activation_type=2)
	model.fit(X, y2indicator(Y))
	print('score:', model.score(Y, model.predict(X)))

	# experiment()
