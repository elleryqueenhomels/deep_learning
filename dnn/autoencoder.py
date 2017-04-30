# AutoEncoder for Unsupervised Learning
# Used for pretraining DNN

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weights, get_activation


class AutoEncoder(object):
	def __init__(self, M, an_id=None, activation_type=1, cost_type=1):
		self.M = M
		self.id = an_id
		self.activation = get_activation(activation_type)
		self.cost_type = cost_type

	def fit(self, X, epochs=1, batch_sz=100, learning_rate=0.5, momentum=0.99, debug=False, print_period=20, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		X = X.astype(np.float32)
		N, D = X.shape

		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N
		n_batches = N // batch_sz

		W = init_weights((D, self.M))
		bh = np.zeros(self.M, dtype=np.float32)
		bo = np.zeros(D, dtype=np.float32)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.bh = theano.shared(bh, 'bh_%s' % self.id)
		self.bo = theano.shared(bo, 'bo_%s' % self.id)
		self.params = [self.W, self.bh, self.bo]
		self.forward_params = [self.W, self.bh]

		# for Momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		X_in = T.fmatrix('X_%s' % self.id)
		X_hat = self.forward_output(X_in)

		# attach it to the object so it can be used later
		FH = self.forward_hidden(X_in)
		self.forward_hidden_op = theano.function(
			inputs=[X_in],
			outputs=FH
		)

		if self.cost_type == 1:
			cost = -T.mean(X_in * T.log(X_hat) + (one - X_in) * T.log(one - X_hat)) # binary cross-entropy
		elif self.cost_type == 2:
			cost = T.mean((X_in - X_hat) * (X_in - X_hat)) # squared error
		else:
			cost = -T.mean(X_in * T.log(X_hat)) # categorical cross-entropy
		cost_op = theano.function(
			inputs=[X_in],
			outputs=cost
		)

		if mu > 0:
			updates = []
			for p, dp in zip(self.params, dparams):
				updates += [
					(p, p + mu*dp - lr*T.grad(cost, p)),
					(dp, mu*dp - lr*T.grad(cost, p))
				]
		else:
			updates = [(p, p - lr*T.grad(cost, p)) for p in self.params]

		train_op = theano.function(
			inputs=[X_in],
			updates=updates
		)

		if debug:
			print('\nTraining AutoEncoder %s:' % self.id)
			costs = []

		for i in range(epochs):
			X = shuffle(X)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]

				train_op(Xbatch)

				if debug:
					if j % print_period == 0:
						the_cost = cost_op(X)
						costs.append(the_cost)
						print('epoch=%d, batch=%d, n_batches=%d: cost=%.6f' % (i, j, n_batches, the_cost))

		if debug:
			the_cost = cost_op(X)
			costs.append(the_cost)
			print('Finally: cost=%.6f' % (the_cost))

			if show_fig:
				plt.plot(costs)
				plt.title('Costs in AutoEncoder %s' % self.id)
				plt.show()

	def forward_hidden(self, X):
		return self.activation(X.dot(self.W) + self.bh)

	def forward_output(self, X):
		Z = self.forward_hidden(X)
		return self.activation(Z.dot(self.W.T) + self.bo)

	def forward(self, X):
		# this 'forward' is the redundancy of 'forward_hidden',
		# just for compatibility.
		return self.activation(X.dot(self.W) + self.bh)

	@staticmethod
	def createFromArray(W, bh, bo, an_id=None, activation_type=1, cost_type=1):
		ae = AutoEncoder(W.shape[1], an_id, activation_type, cost_type)
		ae.W = theano.shared(W.astype(np.float32), 'W_%s' % ae.id)
		ae.bh = theano.shared(bh.astype(np.float32), 'bh_%s' % ae.id)
		ae.bo = theano.shared(bo.astype(np.float32), 'bo_%s' % ae.id)
		ae.params = [ae.W, ae.bh, ae.bo]
		ae.forward_params = [ae.W, ae.bh]
		return ae

