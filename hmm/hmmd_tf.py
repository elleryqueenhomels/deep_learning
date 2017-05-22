# Discrete Hidden Markov Model (HMM) with scaling in TensorFlow using Gradient Descent

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)


class HMM(object):
	def __init__(self, M):
		self.M = M # number of hidden states
		self.session = tf.Session()

	def fit(self, X, V=None, learning_rate=1e-3, max_iter=10, debug=False, print_period=10):
		# train the HMM model using Stochastic Gradient Descent

		# determine V, the vocabulary size (the observation symbol size)
		# assume observables are already integers from 0..V-1
		# X is a jagged array of observed sequences
		if V is None:
			V = max(max(x) for x in X) + 1
		N = len(X)
		M = self.M # for convenience
		if debug:
			print('\nNumber of train samples: %d\n' % N)

		# pi0 = np.ones(M) / M # initial state distribution
		# A0 = random_normalized(M, M) # state transition matrix
		# B0 = random_normalized(M, V) # observation distribution
		pi0 = np.random.randn(M) # initial state distribution
		A0 = np.random.randn(M, M) # state transition matrix
		B0 = np.random.randn(M, V) # observation distribution

		self.set(pi0, A0, B0)

		train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

		init = tf.global_variables_initializer()
		self.session.run(init)
		costs = []
		for it in range(max_iter):
			for n in range(N):
				if debug:
					c = self.get_cost_multi(X).sum()
					costs.append(c)
				self.session.run(train_op, feed_dict={self.tfx: X[n]})

		if debug:
			pi, A, B = self.get_params()
			print('pi:\n', pi)
			print('A:\n', A)
			print('B:\n', B)

			plt.plot(costs)
			plt.title('Costs (log-likelihood)')
			plt.show()

	def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
		M, V = preSoftmaxB.shape

		self.preSoftmaxPi = tf.Variable(preSoftmaxPi.astype(np.float32)) # without normalization
		self.preSoftmaxA = tf.Variable(preSoftmaxA.astype(np.float32)) # without normalization
		self.preSoftmaxB = tf.Variable(preSoftmaxB.astype(np.float32)) # without normalization

		pi = tf.nn.softmax(self.preSoftmaxPi)
		A = tf.nn.softmax(self.preSoftmaxA)
		B = tf.nn.softmax(self.preSoftmaxB)

		# define cost
		self.tfx = tf.placeholder(tf.int32, shape=(None,), name='x')
		def recurrence(old_a_old_s, x_t):
			old_a = tf.reshape(old_a_old_s[0], (1, M))
			a = tf.matmul(old_a, A) * B[:, x_t]
			a = tf.reshape(a, (M,))
			s = tf.reduce_sum(a)
			return a / s, s

		# Remember: TensorFlow scan is going to loop through
		# all the values!
		# we treat the first value differently than the rest
		# so we only want to loop through tfx[1:]
		# the first scale being 1 doesn't affect the log-likelihood
		# because log(1) = 0
		alpha0 = pi * B[:, self.tfx[0]]
		alpha, scale = tf.scan(
			fn=recurrence,
			elems=self.tfx[1:],
			initializer=(alpha0, np.float32(1)),
		)

		self.cost = -tf.reduce_sum(tf.log(scale))

	def reset(self, pi, A, B):
		op1 = self.preSoftmaxPi.assign(pi.astype(np.float32))
		op2 = self.preSoftmaxA.assign(A.astype(np.float32))
		op3 = self.preSoftmaxB.assign(B.astype(np.float32))
		self.session.run([op1, op2, op3])

	def get_cost(self, x):
		# returns -log P(x | model)
		# using the Forward part of the Forward-Backward Algorithm
		return self.session.run(self.cost, feed_dict={self.tfx: x})

	def log_likelihood(self, x):
		return -self.session.run(self.cost, feed_dict={self.tfx: x})

	def get_cost_multi(self, X):
		return np.array([self.get_cost(x) for x in X])

	def get_params(self):
		pi = tf.nn.softmax(self.preSoftmaxPi).eval(session=self.session)
		A = tf.nn.softmax(self.preSoftmaxA).eval(session=self.session)
		B = tf.nn.softmax(self.preSoftmaxB).eval(session=self.session)
		return pi, A, B


# demo
def fit_coin():
	X = []
	for line in open('../data_set/coin_data.txt'):
		# 1 for H, 0 for T
		x = [1 if e == 'H' else 0 for e in line.strip()]
		X.append(x)

	hmm = HMM(2)
	hmm.fit(X, learning_rate=1e-2, max_iter=5, debug=True)

	print('\nFinished training...\n')

	cost = hmm.get_cost_multi(X).sum()
	print('cost with fitted params:', cost)

	# try true values
	pi = np.log(np.array([0.5, 0.5]))
	A = np.log(np.array([[0.1, 0.9], [0.8, 0.2]]))
	B = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
	hmm.reset(pi, A, B)
	cost = hmm.get_cost_multi(X).sum()
	print('cost with true params:', cost, '\n')


if __name__ == '__main__':
	fit_coin()

