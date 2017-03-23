# Artificial Neural Network
import numpy as np


class ANN:
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type

	def fit(self, X, Y, layers=None, activation_type=None, learning_rate=10e-7, epochs=20000):
		if layers != None:
			self.layers = layers
		assert(self.layers != None)
		if activation_type != None:
			self.activation_type = activation_type

		N, D = X.shape
		if len(Y.shape) == 1:
			Y = self.y2indicator(Y)
		elif Y.shape[1] == 1:
			Y = self.y2indicator(np.squeeze(Y))
		K = Y.shape[1]

		self.initialize(D, K)

		# training: Backpropagation
		for i in range(epochs):
			Z = self.forward(X)
			self.backpropagation(Y, Z, learning_rate)
			# for debug:
			if i % 100 == 0:
				print('%d:' % i, 'score =', self.classification_rate(np.argmax(Y, axis=1), np.argmax(Z[-1], axis=1)))

	def initialize(self, D, K):
		self.W = []
		self.b = []

		L = len(self.layers)
		for i in range(L):
			if i == 0:
				W = np.random.randn(D, self.layers[i]) / np.sqrt(D + self.layers[i])
			else:
				W = np.random.randn(self.layers[i-1], self.layers[i]) / np.sqrt(self.layers[i-1] + self.layers[i])
			self.W.append(W)
			self.b.append(np.zeros(self.layers[i]))
		self.W.append(np.random.randn(self.layers[L-1], K) / np.sqrt(self.layers[L-1] + K))
		self.b.append(np.zeros(K))

	def backpropagation(self, T, Z, learning_rate):
		# len(self.W) == len(Z) - 1 == len(self.layers) + 1; len(self.W) == len(self.b)
		delta = Z[-1] - T # Z[-1] is output Y
		for i in reversed(range(len(self.W))):
			self.W[i] -= learning_rate * Z[i].T.dot(delta)
			self.b[i] -= learning_rate * delta.sum(axis=0)
			if self.activation_type == 1:
				delta = delta.dot(self.W[i].T) * (Z[i] * (1 - Z[i]))
			else:
				delta = delta.dot(self.W[i].T) * (1 - Z[i] * Z[i])

	def forward(self, X):
		Z = [X]
		L = len(self.layers) # len(self.layers) == len(self.W) - 1
		for i in range(L):
			Z.append(self.activation(Z[i].dot(self.W[i]) + self.b[i]))
		Z.append(self.softmax(Z[L].dot(self.W[L]) + self.b[L]))
		return Z

	def predict(self, X):
		return np.argmax(self.forward(X)[-1], axis=1)

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)

	def classification_rate(self, Y, P):
		return np.mean(Y == P)

	def activation(self, a):
		if self.activation_type == 1:
			return 1 / (1 + np.exp(-a))
		else:
			return np.tanh(a)

	def softmax(self, a):
		expA = np.exp(a)
		return expA / expA.sum(axis=1, keepdims=True)

	def y2indicator(self, Y):
		N = len(Y)
		K = len(set(Y))
		T = np.zeros((N, K))
		# for i in range(N):
		# 	T[i, int(Y[i])] = 1
		T[np.arange(N), Y.astype(np.int32)] = 1
		return T
