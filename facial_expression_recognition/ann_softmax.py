import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle


class ANN(object):
	def __init__(self, M):
		self.M = M

	# learning rate 10e-6 is too large
	def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		X, Y = X[:-1000], Y[:-1000]

		N, D = X.shape
		K = len(set(Y))
		T = y2indicator(Y)

		self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
		self.b1 = np.random.randn(self.M)
		self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)
		self.b2 = np.random.randn(K)

		costs = []
		best_validation_error = 1
		for i in range(epochs):
			# forward propagation and cost calculation
			pY, Z = self.forward(X)

			# gradient descent step
			pY_T = pY - T
			self.W2 -= learning_rate * (Z.T.dot(pY_T) + reg * self.W2)
			self.b2 -= learning_rate * (pY_T.sum(axis=0) + reg * self.b2)

			# dZ = pY_T.dot(self.W2.T) * (Z > 0) # relu
			dZ = pY_T.dot(self.W2.T) * (1 - Z * Z) # tanh
			self.W1 -= learning_rate * (X.T.dot(dZ) + reg * self.W1)
			self.b1 -= learning_rate * (dZ.sum(axis=0) + reg * self.b1)

			if i % 10 == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = cost2(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
				print('i:', i, 'cost:', c, 'error rate: %.8f%%' % (e * 100))
				if e < best_validation_error:
					best_validation_error = e
		print('\nbest_validation_rate: %.8f%%' % (best_validation_error * 100))

		if show_fig:
			plt.plot(costs, label='Cross-Entropy Cost')
			plt.legend()
			plt.show()

	def forward(self, X):
		# Z = relu(X.dot(self.W1) + self.b1)
		Z = np.tanh(X.dot(self.W1) + self.b1)
		return softmax(Z.dot(self.W2) + self.b2), Z

	def predict(self, X):
		pY, _ = self.forward(X)
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)


def main():
	X, Y = getData()

	model = ANN(200)
	model.fit(X, Y, reg=0, show_fig=True)
	print('\nFinal Score: %.8f%%' % (model.score(X, Y) * 100))


if __name__ == '__main__':
	main()
