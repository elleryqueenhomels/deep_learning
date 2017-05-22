# Continuous-observation HMM in TensorFlow using Gradient Descent

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from generate_continuous import get_signals, simple_init, complex_init

MVN = tf.contrib.distributions.MultivariateNormalDiag


class HMM(object):
	def __init__(self, M, K):
		self.M = M # number of hidden states
		self.K = K # number of Gaussians
		self.session = tf.Session()

	def fit(self, X, learning_rate=1e-3, max_iter=10, debug=False):
		# train the HMM model using Stochastic Gradient Descent

		N = len(X)
		D = X[0].shape[1] # assume each x is organized (T, D)
		self.D = D
		M = self.M # for convenience
		K = self.K # for convenience
		if debug:
			print('\nnumber of train samples: %d\n' % N)

		pi0 = np.random.randn(M) # initial state distribution
		A0 = np.random.randn(M, M) # state transition matrix
		R0 = np.random.randn(M, K) # mixture proportions
		mu0 = np.zeros((M, K, D))
		for j in range(M):
			for k in range(K):
				random_idx = np.random.choice(N)
				x = X[random_idx]
				random_time_idx = np.random.choice(len(x))
				mu0[j,k] = x[random_time_idx]
		sigma0 = np.random.randn(M, K, D)

		self.set(pi0, A0, R0, mu0, sigma0)

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
			pi, A, R, mu, sigma = self.get_params()
			print('pi:\n', pi)
			print('A:\n', A)
			print('R:\n', R)
			print('mu:\n', mu)
			print('sigma:\n', sigma)

			plt.plot(costs)
			plt.title('Costs (log-likelihood)')
			plt.show()

	def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, logSigma):
		self.preSoftmaxPi = tf.Variable(preSoftmaxPi.astype(np.float32))
		self.preSoftmaxA = tf.Variable(preSoftmaxA.astype(np.float32))
		self.preSoftmaxR = tf.Variable(preSoftmaxR.astype(np.float32))
		self.mu = tf.Variable(mu.astype(np.float32))
		self.logSigma = tf.Variable(logSigma.astype(np.float32))

		pi = tf.nn.softmax(self.preSoftmaxPi)
		A = tf.nn.softmax(self.preSoftmaxA)
		R = tf.nn.softmax(self.preSoftmaxR)
		sigma = tf.exp(self.logSigma)

		# tfx: (T, D)
		self.tfx = tf.placeholder(tf.float32, shape=(None, self.D), name='x')

		# first we need to calculate B
		# B[j,t] = probability of x being in state j at time t
		#        = Gaussian mixture P( x(t) | mu(j), sigma(j) )
		# idea: first calculate components and sum
		# note: we can use a for loop because M and K are not TF variables
		self.mvns = []
		for j in range(self.M):
			self.mvns.append([])
			for k in range(self.K):
				self.mvns[j].append(
					MVN(self.mu[j,k], sigma[j,k])
				)

		# note: we can use a for loop because M and K are not TF variables
		B = []
		for j in range(self.M):
			components = []
			for k in range(self.K):
				components.append(
					self.mvns[j][k].prob(self.tfx)
				)

			# why?
			# because we can stack a list of tensors
			# but not a list of lists of tensors

			# components[j] will be (K, T)
			# we now want to multiply by the mixture probability (R)
			# result is (M, T)
			# which gives us P( x(t) | state j )
			components = tf.stack(components) # shape: (K, T)
			R_j = tf.reshape(R[j], [1, self.K]) # shape: (1, K)
			p_x_t_j = tf.matmul(R_j, components) # shape: (1, T)
			components = tf.reshape(p_x_t_j, [-1]) # shape: (T, )
			B.append(components)

		# B should now be (M, T)
		B = tf.stack(B)

		# we should make it (T, M) since tf.scan will loop through the first index
		B = tf.transpose(B, [1, 0])

		# now perform the Forward Algorithm
		def recurrence(old_a_old_s, B_t):
			old_a = tf.reshape(old_a_old_s[0], [1, self.M])
			a = tf.matmul(old_a, A) * B_t
			a = tf.reshape(a, [self.M,])
			s = tf.reduce_sum(a)
			return a / s, s

		alpha, scale = tf.scan(
			fn=recurrence,
			elems=B,
			initializer=(pi * B[0], np.float32(1)),
		)

		self.cost = -tf.reduce_sum(tf.log(scale))

	def reset(self, pi, A, R, mu, sigma):
		op1 = self.preSoftmaxPi.assign(pi.astype(np.float32))
		op2 = self.preSoftmaxA.assign(A.astype(np.float32))
		op3 = self.preSoftmaxR.assign(R.astype(np.float32))
		op4 = self.mu.assign(mu.astype(np.float32))
		op5 = self.logSigma.assign(sigma.astype(np.float32))
		self.session.run([op1, op2, op3, op4, op5])

	def get_cost(self, x):
		return self.session.run(self.cost, feed_dict={self.tfx: x})

	def get_cost_multi(self, X):
		return np.array([self.get_cost(x) for x in X])

	def get_params(self):
		pi = tf.nn.softmax(self.preSoftmaxPi).eval(session=self.session)
		A = tf.nn.softmax(self.preSoftmaxA).eval(session=self.session)
		R = tf.nn.softmax(self.preSoftmaxR).eval(session=self.session)
		mu = self.mu.eval(session=self.session)
		sigma = tf.exp(self.logSigma).eval(session=self.session)
		return pi, A, R, mu, sigma


def real_signal():
	import wave
	spf = wave.open('../data_set/helloworld.wav', 'r')

	# Extract Raw Audio from Wav File
	# sampling rate = 16000 Hz
	# bits per sample = 16
	# The first is quantization in time
	# The second is quantization in amplitude
	# We also do this for images!
	# 2^16 = 65536 is how many different sound levels we have
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, dtype=np.int16)
	T = len(signal)
	signal = (signal - signal.mean()) / signal.std()

	hmm = HMM(5, 3)
	# signal needs to be of shape N x T(n) x D
	hmm.fit(signal.reshape(1, T, 1), learning_rate=1e-5, max_iter=20, debug=True)

	print('\nFinished training...\n')

	cost = hmm.get_cost(signal.reshape(T, 1))
	print('cost for fitted params:', cost, '\n')


def fake_signal(N=10, T=20, init=complex_init):
	signals = get_signals(N=N, T=T, init=init)
	for signal in signals:
		for d in range(signal.shape[1]):
			plt.plot(signal[:,d])
	plt.show()

	hmm = HMM(5, 3)
	hmm.fit(signals, debug=True)

	print('\nFinished training...\n')

	cost = hmm.get_cost_multi(signals).sum()
	print('cost for fitted params:', cost)

	# test in actual params
	_, _, _, pi, A, R, mu, sigma = init()
	if np.any(pi == 0):
		pi += 1e-3
		pi /= pi.sum()
	pi = np.log(pi)
	A = np.log(A)
	R = np.log(R)
	M, K, D, _ = sigma.shape # need to convert full cov into diag cov
	logSigma = np.zeros((M, K, D))
	for j in range(M):
		for k in range(K):
			logSigma[j,k] = np.log(np.diag(sigma[j,k]))

	hmm.reset(pi, A, R, mu, logSigma)
	cost = hmm.get_cost_multi(signals).sum()
	print('cost for actual params:', cost, '\n')


if __name__ == '__main__':
	# real_signal()
	fake_signal(init=complex_init)

