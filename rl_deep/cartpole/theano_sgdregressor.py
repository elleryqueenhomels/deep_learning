# Use Theano to re-implement the SGDRegressor

import numpy as np
import theano
import theano.tensor as T
import q_learning


class SGDRegressor:
	def __init__(self, D, learning_rate=0.1):
		print('Use Theano to implement SGDRegressor')
		w = np.random.randn(D) / np.sqrt(D)
		self.w = theano.shared(w)
		self.lr = learning_rate

		X = T.matrix('X')
		Y = T.vector('Y')
		Y_hat = X.dot(self.w)
		delta = Y - Y_hat
		cost = delta.dot(delta)
		grad = T.grad(cost, self.w)
		updates = [(self.w, self.w - self.lr * grad)]

		self.train_op = theano.function(
			inputs=[X, Y],
			updates=updates,
		)

		self.predict_op = theano.function(
			inputs=[X],
			outputs=Y_hat,
		)

	def partial_fit(self, X, Y):
		self.train_op(X, Y)

	def predict(self, X):
		return self.predict_op(X)


if __name__ == '__main__':
	q_learning.SGDRegressor = SGDRegressor
	q_learning.main()

