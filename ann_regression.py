# Artificial Neural Network for Regression, With: mini-batch, RMSprop(adaptive learning rate), Nesterov Momentum, regularization(L1 & L2)
import numpy as np


class ANN_Regression(object):
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type


	def fit(self, X, Y, layers=None, activation_type=None, epochs=10000, batch_size=0, learning_rate=10e-5, decay=0, momentum=0, regularization1=0, regularization2=0):
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

		if batch_size > 0 and batch_size < N:
			# training: Backpropagation, using batch gradient descent
			n_batches = int(N / batch_size)
			for i in range(epochs):
				for j in range(n_batches):
					if j == n_batches - 1:
						Xbatch = X[j*batch_size:]
						Ybatch = Y[j*batch_size:]
					else:
						Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
						Ybatch = Y[j*batch_size:(j*batch_size + batch_size)]

					# forward propagation
					Z = self.forward(Xbatch)

					# gradient descent step
					self.backpropagation(Ybatch, Z, learning_rate, decay, momentum, regularization1, regularization2)

					# for debug:
					if i % 100 == 0:
						score = self.r_squared(Ybatch, Z[-1])
						print('iteration=%d, batch=%d:' % (i, j), 'score = %.8f%%' % (score * 100))
		else:
			# training: Backpropagation, using full gradient descent
			for i in range(epochs):
				# forward propagation
				Z = self.forward(X)

				# gradient descent step
				self.backpropagation(Y, Z, learning_rate, decay, momentum, regularization1, regularization2)

				# for debug:
				if i % 100 == 0:
					score = self.r_squared(Y, Z[-1])
					print('iteration=%d:' % i, 'score = %.8f%%' % (score * 100))


	def initialize(self, D, K):
		self.W = []
		self.b = []
		self.cache_W = [] # for RMSprop adaptive learning rate
		self.cache_b = [] # for RMSprop adaptive learning rate
		self.dW = [] # for Nesterov Momentum
		self.db = [] # for Nesterov Momentum

		for i in range(len(self.layers)):
			if i == 0:
				W = np.random.randn(D, self.layers[i]) / np.sqrt(D + self.layers[i])
			else:
				W = np.random.randn(self.layers[i-1], self.layers[i]) / np.sqrt(self.layers[i-1] + self.layers[i])
			self.W.append(W)
			self.b.append(np.zeros(self.layers[i]))
			self.cache_W.append(np.zeros(W.shape))
			self.cache_b.append(np.zeros(self.layers[i]))
			self.dW.append(np.zeros(W.shape))
			self.db.append(np.zeros(self.layers[i]))
		self.W.append(np.random.randn(self.layers[-1], K) / np.sqrt(self.layers[-1] + K))
		self.b.append(np.zeros(K))
		self.cache_W.append(np.zeros((self.layers[-1], K)))
		self.cache_b.append(np.zeros(K))
		self.dW.append(np.zeros((self.layers[-1], K)))
		self.db.append(np.zeros(K))


	def backpropagation(self, T, Z, learning_rate, decay, momentum, regularization1, regularization2):
		# len(self.W) == len(Z) - 1 == len(self.layers) + 1; len(self.W) == len(self.b)

		delta = Z[-1] - T # Z[-1] is output Y

		# Optional: RMSprop(adaptive learning rate), Nesterov Momentum
		if decay > 0 and decay < 1:
			eps = 1e-10
			if momentum > 0:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + regularization1 * np.sign(self.W[i]) + regularization2 * self.W[i]
					self.cache_W[i] = decay*self.cache_W[i] + (1 - decay)*gradW*gradW
					self.dW[i] = momentum*momentum*self.dW[i] - (1 + momentum)*learning_rate*gradW/(np.sqrt(self.cache_W[i]) + eps)
					self.W[i] += self.dW[i]

					gradb = delta.sum(axis=0) + regularization1 * np.sign(self.b[i]) + regularization2 * self.b[i]
					self.cache_b[i] = decay*self.cache_b[i] + (1 - decay)*gradb*gradb
					self.db[i] = momentum*momentum*self.db[i] - (1 + momentum)*learning_rate*gradb/(np.sqrt(self.cache_b[i]) + eps)
					self.b[i] += self.db[i]

					delta = self.get_delta(delta, self.W[i], Z[i])
			else:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + regularization1 * np.sign(self.W[i]) + regularization2 * self.W[i]
					self.cache_W[i] = decay*self.cache_W[i] + (1 - decay)*gradW*gradW
					self.W[i] -= learning_rate * gradW / (np.sqrt(self.cache_W[i]) + eps)

					gradb = delta.sum(axis=0) + regularization1 * np.sign(self.b[i]) + regularization2 * self.b[i]
					self.cache_b[i] = decay*self.cache_b[i] + (1 - decay)*gradb*gradb
					self.b[i] -= learning_rate * gradb / (np.sqrt(self.cache_b[i]) + eps)

					delta = self.get_delta(delta, self.W[i], Z[i])
		else:
			if momentum > 0:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + regularization1 * np.sign(self.W[i]) + regularization2 * self.W[i]
					self.dW[i] = momentum*momentum*self.dW[i] - (1 + momentum)*learning_rate*gradW
					self.W[i] += self.dW[i]

					gradb = delta.sum(axis=0) + regularization1 * np.sign(self.b[i]) + regularization2 * self.b[i]
					self.db[i] = momentum*momentum*self.db[i] - (1 + momentum)*learning_rate*gradb
					self.b[i] += self.db[i]

					delta = self.get_delta(delta, self.W[i], Z[i])
			else:
				for i in reversed(range(len(self.W))):
					self.W[i] -= learning_rate * (Z[i].T.dot(delta) + regularization1 * np.sign(self.W[i]) + regularization2 * self.W[i])
					self.b[i] -= learning_rate * (delta.sum(axis=0) + regularization1 * np.sign(self.b[i]) + regularization2 * self.b[i])
					delta = self.get_delta(delta, self.W[i], Z[i])


	def get_delta(self, delta, W, Z):
		if self.activation_type == 1:
			delta = delta.dot(W.T) * (1 - Z * Z)
		elif self.activation_type == 2:
			delta = delta.dot(W.T) * (Z > 0)
		else:
			delta = delta.dot(W.T) * Z * (1 - Z)
		return delta


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


	def cost(self, Y, Yhat):
		diff = Y - Yhat
		return np.sum(diff * diff)


	def activation(self, a):
		if self.activation_type == 1:
			return np.tanh(a)
		elif self.activation_type == 2:
			return a * (a > 0)
		else:
			return 1 / (1 + np.exp(-a))
