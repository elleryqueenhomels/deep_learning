# Artificial Neural Network for Regression
import numpy as np


class ANN_Regression:
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type

	def fit(self, X, Y, layers=None, activation_type=None, learning_rate=10e-7, epochs=20000, regularization1=0, regularization2=0):
		if layers != None:
			self.layers = layers
		assert(self.layers != None)
		if activation_type != None:
			self.activation_type = activation_type

		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		N, D = X.shape
		if len(Y.shape) == 1:
			Y = Y.reshape(-1, 1)
		K = Y.shape[1]

		self.initialize(D, K)

		# training: Backpropagation
		for i in range(epochs):
			Z = self.forward(X)
			self.backpropagation(Y, Z, learning_rate, regularization1, regularization2)
			# for debug:
			if i % 100 == 0:
				score = self.r_squared(Y, Z[-1])
				print('%d:' % i, 'score = %.8f%%' % (score * 100))

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

	def backpropagation(self, T, Z, learning_rate, regularization1, regularization2):
		# len(self.W) == len(Z) - 1 == len(self.layers) + 1; len(self.W) == len(self.b)
		delta = Z[-1] - T # Z[-1] is output Y
		for i in reversed(range(len(self.W))):
			self.W[i] -= (learning_rate * Z[i].T.dot(delta) + regularization1 * np.sign(self.W[i]) + regularization2 * self.W[i])
			self.b[i] -= (learning_rate * delta.sum(axis=0) + regularization1 * np.sign(self.b[i]) + regularization2 * self.b[i])
			if self.activation_type == 1:
				delta = delta.dot(self.W[i].T) * (1 - Z[i] * Z[i])
			else:
				delta = delta.dot(self.W[i].T) * (Z[i] * (1 - Z[i]))

	def forward(self, X):
		Z = [X]
		L = len(self.layers) # len(self.layers) == len(self.W) - 1
		for i in range(L):
			Z.append(self.activation(Z[i].dot(self.W[i]) + self.b[i]))
		Z.append(Z[L].dot(self.W[L]) + self.b[L])
		return Z

	def predict(self, X):
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		return np.squeeze(self.forward(X)[-1])

	def score(self, X, Y):
		Yhat = self.predict(X)
		return self.r_squared(Y, Yhat)

	def r_squared(self, Y, Yhat):
		Y = np.squeeze(Y)
		Yhat = np.squeeze(Yhat)
		Y1 = Y - Yhat
		Y2 = Y - Y.mean()
		return 1 - Y1.dot(Y1) / Y2.dot(Y2)

	def activation(self, a):
		if self.activation_type == 1:
			return np.tanh(a)
		else:
			return 1 / (1 + np.exp(-a))
