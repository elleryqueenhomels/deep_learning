# Restricted Boltzmann Machine for Unsupervised Learning
# Used for pretraining DNN
# Deep stacked RBMs have a special name: Deep Belief Network (DBN)

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import init_weights, get_activation


class RBM(object):
	def __init__(self, M, an_id=None, activation_type=1, cost_type=1):
		self.M = M
		self.id = an_id
		self.rng = RandomStreams()
		self.activation = get_activation(activation_type)
		self.cost_type = cost_type

	def fit(self, X, epochs=1, batch_sz=100, learning_rate=0.5, momentum=0, debug=False, print_period=20, show_fig=False):
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
		c = np.zeros(self.M, dtype=np.float32)
		b = np.zeros(D, dtype=np.float32)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.c = theano.shared(c, 'c_%s' % self.id)
		self.b = theano.shared(b, 'b_%s' % self.id)
		self.params = [self.W, self.c, self.b]
		self.forward_params = [self.W, self.c]

		# for momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		X_in = T.fmatrix('X_%s' % self.id)

		# attach it to the object so it can be used later
		FH = self.forward_hidden(X_in)
		self.forward_hidden_op = theano.function(
			inputs=[X_in],
			outputs=FH
		)

		# we don't use this cost to do any updates/training
		# but we would like to see how this cost function changes
		# as we do Contrstive Divergence
		if debug:
			X_hat = self.forward_output(X_in)
			cost = -T.mean(X_in * T.log(X_hat) + (one - X_in) * T.log(one - X_hat)) # binary cross-entropy
			cost_op = theano.function(inputs=[X_in], outputs=cost)

		# Now begin to do Contrastive Divergence (CD-1)
		# do one round of Gibbs Sampling to obtain X_sample
		H = self.sample_h_given_v(X_in)
		X_sample = self.sample_v_given_h(H)

		# define the objective function, updates, and train operation
		objective = T.mean(self.free_energy(X_in)) - T.mean(self.free_energy(X_sample))

		# need to consider X_sample as constant because Theano can't take the derivative of random numbers.
		if mu > 0:
			updates = []
			for p, dp in zip(self.params, dparams):
				updates += [
					(p, p + mu*dp - lr*T.grad(objective, p, consider_constant=[X_sample])),
					(dp, mu*dp - lr*T.grad(objective, p, consider_constant=[X_sample]))
				]
		else:
			updates = [(p, p - lr*T.grad(objective, p, consider_constant=[X_sample])) for p in self.params]

		train_op = theano.function(
			inputs=[X_in],
			updates=updates
		)

		if debug:
			print('\nTraining Restricted Boltzmann Machine %s:' % self.id)
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
				plt.title('Costs in Restricted Boltzmann Machine %s' % self.id)
				plt.show()

	def free_energy(self, V):
		one = np.float32(1)
		return -V.dot(self.b) - T.sum(T.log(one + T.exp(self.c + V.dot(self.W))), axis=1)

	def sample_h_given_v(self, V):
		p_h_given_v = self.activation(V.dot(self.W) + self.c)
		h_sample = self.rng.binomial(n=1, p=p_h_given_v, size=p_h_given_v.shape, dtype='float32')
		return h_sample

	def sample_v_given_h(self, H):
		p_v_given_h = self.activation(H.dot(self.W.T) + self.b)
		v_sample = self.rng.binomial(n=1, p=p_v_given_h, size=p_v_given_h.shape, dtype='float32')
		return v_sample

	def forward_hidden(self, X):
		return self.activation(X.dot(self.W) + self.c)

	def forward_output(self, X):
		Z = self.forward_hidden(X)
		return self.activation(Z.dot(self.W.T) + self.b)

	def forward(self, X):
		# this 'forward' is the redundancy of 'forward_hidden',
		# just for compatibility.
		return self.activation(X.dot(self.W) + self.c)

	@staticmethod
	def createFromArray(W, c, b, an_id=None, activation_type=1, cost_type=1):
		rbm = RBM(W.shape[1], an_id, activation_type, cost_type)
		rbm.W = theano.shared(W.astype(np.float32), 'W_%s' % rbm.id)
		rbm.c = theano.shared(c.astype(np.float32), 'c_%s' % rbm.id)
		rbm.b = theano.shared(b.astype(np.float32), 'b_%s' % rbm.id)
		rbm.params = [rbm.W, rbm.c, rbm.b]
		rbm.forward_params = [rbm.W, rbm.c]
		return rbm

