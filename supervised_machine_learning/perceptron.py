# This is an example of Perceptron -- a Linear Binary Classifier.

import numpy as np
import matplotlib.pyplot as plt
from util import get_data as get_mnist
from util import getFacialData
from datetime import datetime


debug = False

def get_data():
	w = np.array([-0.5, 0.5])
	b = 0.1
	# np.random.random(): uniformly distributed in [0, 1]
	X = np.random.random((1000, 2))*2 - 1
	Y = np.sign(X.dot(w) + b)
	return X, Y

def get_simple_xor():
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	Y = np.array([0, 1, 1, 0])
	return X, Y


class Perceptron:
	def fit(self, X, Y, learning_rate=1.0, epochs=50000):
		# The solution after using function get_data():
		# self.w = np.array([-0.5, 0.5])
		# self.b = 0.1

		# initialize random weights
		N, D = X.shape
		self.w = np.random.randn(D)
		self.b = 0

		costs = []
		for epoch in range(epochs):
			# determine which samples are misclassified, if any.
			Yhat = self.predict(X)
			incorrect = np.nonzero(Yhat != Y)[0]

			# cost is incorrect rate
			c = len(incorrect) / float(N)
			costs.append(c)

			if len(incorrect) == 0:
				break

			# randomly choose an incorrect sample
			# i = np.random.choice(incorrect)
			# self.w += learning_rate*Y[i]*X[i]
			# self.b += learning_rate*Y[i]

			# gradient descent
			for i in incorrect:
				self.w += learning_rate*Y[i]*X[i]
				self.b += learning_rate*Y[i]

		print('\nFinal w:', self.w, '; Final b:', self.b, '; epochs:', epoch+1, '/', epochs)
		if debug:
			plt.plot(costs)
			plt.title('Cost')
			plt.show()

	def predict(self, X):
		return np.sign(X.dot(self.w) + self.b)

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)


if __name__ == '__main__':
	# linearly separable data
	# X, Y = get_data()
	# if debug:
	# 	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
	# 	plt.show()
	# Ntrain = int(len(Y) / 2)
	# Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	# Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# model = Perceptron()
	# t0 = datetime.now()
	# model.fit(Xtrain, Ytrain)
	# print('\nTraining time:', (datetime.now() - t0))

	# t0 = datetime.now()
	# print('\nTrain accuracy:', model.score(Xtrain, Ytrain))
	# print('Train accuracy time:', (datetime.now() - t0), ' Train size:', len(Ytrain))

	# t0 = datetime.now()
	# print('\nTest accuracy:', model.score(Xtest, Ytest))
	# print('Test accuracy time:', (datetime.now() - t0), ' Test size:', len(Ytest))

	# MNIST data or Facial Expression Recognition data
	t0 = datetime.now()
	# X, Y = get_mnist()
	X, Y = getFacialData()
	print('\nGet data time:', (datetime.now() - t0))
	idx = np.logical_or(Y == 2, Y == 5)
	X = X[idx]
	Y = Y[idx]
	Y[Y == 2] = -1
	Y[Y == 5] = 1

	Ntrain = int(len(Y) / 2)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	model = Perceptron()
	t0 = datetime.now()
	model.fit(Xtrain, Ytrain, learning_rate=10e-3)
	print('\nTraining time:', (datetime.now() - t0))

	t0 = datetime.now()
	print('\nTrain accuracy:', model.score(Xtrain, Ytrain))
	print('Train accuracy time:', (datetime.now() - t0), ' Train size:', len(Ytrain))

	t0 = datetime.now()
	print('\nTest accuracy:', model.score(Xtest, Ytest))
	print('Test accuracy time:', (datetime.now() - t0), ' Test size:', len(Ytest))

	# XOR data
	# print('\nXOR result:')
	# X, Y = get_simple_xor()
	# model.fit(X, Y)
	# print('XOR accuracy:', model.score(X, Y))