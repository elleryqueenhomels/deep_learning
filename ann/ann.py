# Artificial Neural Network for Classification, With: mini-batch, RMSprop(adaptive learning rate), Nesterov Momentum, regularization(L1 & L2)
import numpy as np


class ANN(object):
	def __init__(self, layers=None, activation_type=1):
		self.layers = layers
		self.activation_type = activation_type


	def fit(self, X, Y, epochs=10000, batch_sz=0, learning_rate=10e-5, decay=0, momentum=0, reg_l1=0, reg_l2=0, debug=False, cal_train=False, debug_points=100, valid_set=None):
		assert(self.layers is not None)

		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		if len(Y.shape) == 1:
			Y = self.y2indicator(Y)
		elif Y.shape[1] == 1:
			Y = self.y2indicator(np.squeeze(Y))

		# for debug: pre-process training set and validation set
		if debug:
			if cal_train:
				Ytrain = np.argmax(Y, axis=1)
			if valid_set is not None:
				if len(valid_set) < 2 or len(valid_set[0]) != len(valid_set[1]):
					valid_set = None
				else:
					if len(valid_set[0].shape) == 1:
						valid_set[0] = valid_set[0].reshape(-1, 1)
					if valid_set[0].shape[1] != X.shape[1]:
						valid_set = None
					else:
						Xvalid, Yvalid = valid_set[0], np.squeeze(valid_set[1])
						if len(Yvalid.shape) == 2:
							Yvalid = np.argmax(Yvalid, axis=1)
			debug = cal_train or (valid_set is not None)

		N, D = X.shape
		K = Y.shape[1]

		self.initialize(D, K)

		if debug:
			costs_train, costs_valid = [], []
			scores_train, scores_valid = [], []

		if batch_sz > 0 and batch_sz < N:
			# training: Backpropagation, using batch gradient descent
			n_batches = int(N / batch_sz)
			if debug:
				debug_points = np.sqrt(debug_points)
				print_epoch, print_batch = max(int(epochs / debug_points), 1), max(int(n_batches / debug_points), 1)

			for i in range(epochs):
				idx = np.arange(N)
				np.random.shuffle(idx)
				for j in range(n_batches):
					batch_idx = idx[j*batch_sz:(j*batch_sz + batch_sz)]
					Xbatch = X[batch_idx]
					Ybatch = Y[batch_idx]

					# forward propagation
					Z = self.forward(Xbatch)

					# gradient descent step
					self.backpropagation(Ybatch, Z, learning_rate, decay, momentum, reg_l1, reg_l2)

					# for debug:
					if debug:
						if i % print_epoch == 0 and j % print_batch == 0:
							if cal_train:
								pYtrain = self.forward(X)[-1]
								ctrain = self.cost(Ytrain, pYtrain)
								strain = self.classification_rate(Ytrain, np.argmax(pYtrain, axis=1))
								costs_train.append(ctrain)
								scores_train.append(strain)
								print('epoch=%d, batch=%d, n_batches=%d: cost_train=%s, score_train=%.6f%%' % (i, j, n_batches, ctrain, strain*100))
							if valid_set is not None:
								pYvalid = self.forward(Xvalid)[-1]
								cvalid = self.cost(Yvalid, pYvalid)
								svalid = self.classification_rate(Yvalid, np.argmax(pYvalid, axis=1))
								costs_valid.append(cvalid)
								scores_valid.append(svalid)
								print('epoch=%d, batch=%d, n_batches=%d: cost_valid=%s, score_valid=%.6f%%' % (i, j, n_batches, cvalid, svalid*100))
		else:
			# training: Backpropagation, using full gradient descent
			if debug:
				print_epoch = max(int(epochs / debug_points), 1)

			for i in range(epochs):
				# forward propagation
				Z = self.forward(X)

				# gradient descent step
				self.backpropagation(Y, Z, learning_rate, decay, momentum, reg_l1, reg_l2)

				# for debug:
				if debug:
					if i % print_epoch == 0:
						if cal_train:
							pYtrain = self.forward(X)[-1]
							ctrain = self.cost(Ytrain, pYtrain)
							strain = self.classification_rate(Ytrain, np.argmax(pYtrain, axis=1))
							costs_train.append(ctrain)
							scores_train.append(strain)
							print('epoch=%d: cost_train=%s, score_train=%.6f%%' % (i, ctrain, strain*100))
						if valid_set is not None:
							pYvalid = self.forward(Xvalid)[-1]
							cvalid = self.cost(Yvalid, pYvalid)
							svalid = self.classification_rate(Yvalid, np.argmax(pYvalid, axis=1))
							costs_valid.append(cvalid)
							scores_valid.append(svalid)
							print('epoch=%d: cost_valid=%s, score_valid=%.6f%%' % (i, cvalid, svalid*100))

		if debug:
			if cal_train:
				pYtrain = self.forward(X)[-1]
				ctrain = self.cost(Ytrain, pYtrain)
				strain = self.classification_rate(Ytrain, np.argmax(pYtrain, axis=1))
				costs_train.append(ctrain)
				scores_train.append(strain)
				print('Final validation: cost_train=%s, score_train=%.6f%%, train_size=%d' % (ctrain, strain*100, len(Ytrain)))
			if valid_set is not None:
				pYvalid = self.forward(Xvalid)[-1]
				cvalid = self.cost(Yvalid, pYvalid)
				svalid = self.classification_rate(Yvalid, np.argmax(pYvalid, axis=1))
				costs_valid.append(cvalid)
				scores_valid.append(svalid)
				print('Final validation: cost_valid=%s, score_valid=%.6f%%, valid_size=%d' % (cvalid, svalid*100, len(Yvalid)))

			import matplotlib.pyplot as plt
			if cal_train:
				plt.plot(costs_train, label='training set')
			if valid_set is not None:
				plt.plot(costs_valid, label='validation set')
			plt.title('Cross-Entropy Cost')
			plt.legend()
			plt.show()
			if cal_train:
				plt.plot(scores_train, label='training set')
			if valid_set is not None:
				plt.plot(scores_valid, label='validation set')
			plt.title('Classification Rate')
			plt.legend()
			plt.show()


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


	def backpropagation(self, T, Z, learning_rate, decay, momentum, reg_l1, reg_l2):
		# len(self.W) == len(Z) - 1 == len(self.layers) + 1; len(self.W) == len(self.b)

		delta = Z[-1] - T # Z[-1] is output Y

		# Optional: RMSprop(adaptive learning rate), Nesterov Momentum
		if decay > 0 and decay < 1:
			eps = 1e-10
			if momentum > 0:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + reg_l1 * np.sign(self.W[i]) + reg_l2 * self.W[i]
					self.cache_W[i] = decay*self.cache_W[i] + (1 - decay)*gradW*gradW
					self.dW[i] = momentum*momentum*self.dW[i] - (1 + momentum)*learning_rate*gradW/(np.sqrt(self.cache_W[i]) + eps)
					self.W[i] += self.dW[i]

					gradb = delta.sum(axis=0) + reg_l1 * np.sign(self.b[i]) + reg_l2 * self.b[i]
					self.cache_b[i] = decay*self.cache_b[i] + (1 - decay)*gradb*gradb
					self.db[i] = momentum*momentum*self.db[i] - (1 + momentum)*learning_rate*gradb/(np.sqrt(self.cache_b[i]) + eps)
					self.b[i] += self.db[i]

					delta = self.get_delta(delta, self.W[i], Z[i])
			else:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + reg_l1 * np.sign(self.W[i]) + reg_l2 * self.W[i]
					self.cache_W[i] = decay*self.cache_W[i] + (1 - decay)*gradW*gradW
					self.W[i] -= learning_rate * gradW / (np.sqrt(self.cache_W[i]) + eps)

					gradb = delta.sum(axis=0) + reg_l1 * np.sign(self.b[i]) + reg_l2 * self.b[i]
					self.cache_b[i] = decay*self.cache_b[i] + (1 - decay)*gradb*gradb
					self.b[i] -= learning_rate * gradb / (np.sqrt(self.cache_b[i]) + eps)

					delta = self.get_delta(delta, self.W[i], Z[i])
		else:
			if momentum > 0:
				for i in reversed(range(len(self.W))):
					gradW = Z[i].T.dot(delta) + reg_l1 * np.sign(self.W[i]) + reg_l2 * self.W[i]
					self.dW[i] = momentum*momentum*self.dW[i] - (1 + momentum)*learning_rate*gradW
					self.W[i] += self.dW[i]

					gradb = delta.sum(axis=0) + reg_l1 * np.sign(self.b[i]) + reg_l2 * self.b[i]
					self.db[i] = momentum*momentum*self.db[i] - (1 + momentum)*learning_rate*gradb
					self.b[i] += self.db[i]

					delta = self.get_delta(delta, self.W[i], Z[i])
			else:
				for i in reversed(range(len(self.W))):
					self.W[i] -= learning_rate * (Z[i].T.dot(delta) + reg_l1 * np.sign(self.W[i]) + reg_l2 * self.W[i])
					self.b[i] -= learning_rate * (delta.sum(axis=0) + reg_l1 * np.sign(self.b[i]) + reg_l2 * self.b[i])
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
	
	
	def cost(self, T, Y):
		return -np.log(Y[np.arange(len(T)), T.astype(np.int32)]).mean()


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


	def y2indicator(self, Y, K=None):
		N = len(Y)
		if K == None:
			K = len(set(Y))
		T = np.zeros((N, K))
		# for i in range(N):
		# 	T[i, int(Y[i])] = 1
		T[np.arange(N), Y.astype(np.int32)] = 1
		return T

