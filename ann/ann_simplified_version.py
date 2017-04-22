# Artificial Neural Network for Classification (a simplified version)
# Using full gradient descent, NO adaptive learning rate & momentum
# But still has regularization (L1 & L2)
import numpy as np


class ANN(object):
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type

	def fit(self, X, Y, layers=None, activation_type=None, learning_rate=10e-5, epochs=20000, reg_l1=0, reg_l2=0):
		if layers != None:
			self.layers = layers
		assert(self.layers != None)
		if activation_type != None:
			self.activation_type = activation_type
		
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		N, D = X.shape
		if len(Y.shape) == 1:
			Y = self.y2indicator(Y)
		elif Y.shape[1] == 1:
			Y = self.y2indicator(np.squeeze(Y))
		K = Y.shape[1]

		self.initialize(D, K)

		# training: Backpropagation
		for i in range(epochs):
			# forward propagation
			Z = self.forward(X)
			
			# gradient descent step
			self.backpropagation(Y, Z, learning_rate, reg_l1, reg_l2)
			
			# for debug:
			if i % 100 == 0:
				score = self.classification_rate(np.argmax(Y, axis=1), np.argmax(Z[-1], axis=1))
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

	def backpropagation(self, T, Z, learning_rate, reg_l1, reg_l2):
		# len(self.W) == len(Z) - 1 == len(self.layers) + 1; len(self.W) == len(self.b)
		delta = Z[-1] - T # Z[-1] is output Y
		for i in reversed(range(len(self.W))):
			self.W[i] -= learning_rate * (Z[i].T.dot(delta) + reg_l1 * np.sign(self.W[i]) + reg_l2 * self.W[i])
			self.b[i] -= learning_rate * (delta.sum(axis=0) + reg_l1 * np.sign(self.b[i]) + reg_l2 * self.b[i])
			if self.activation_type == 1:
				delta = delta.dot(self.W[i].T) * (1 - Z[i] * Z[i])
			elif self.activation_type == 2:
				delta = delta.dot(self.W[i].T) * (Z[i] > 0)
			else:
				delta = delta.dot(self.W[i].T) * (Z[i] * (1 - Z[i]))

	def forward(self, X):
		Z = [X]
		L = len(self.layers) # len(self.layers) == len(self.W) - 1
		for i in range(L):
			Z.append(self.activation(Z[i].dot(self.W[i]) + self.b[i]))
		Z.append(self.softmax(Z[L].dot(self.W[L]) + self.b[L]))
		return Z

	def predict(self, X):
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		return np.argmax(self.forward(X)[-1], axis=1)

	def score(self, X, Y):
		P = self.predict(X)
		return self.classification_rate(Y, P)

	def classification_rate(self, Y, P):
		Y = np.squeeze(Y)
		P = np.squeeze(P)
		return np.mean(Y == P)

	def activation(self, a):
		if self.activation_type == 1:
			return np.tanh(a)
		elif self.activation_type == 2:
			return a * (a > 0)
		else:
			return 1 / (1 + np.exp(-a))

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

