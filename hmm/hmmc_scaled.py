# Continuous-observation HMM with scaling and multiple observations (treated as concatenated sequence)

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn
from generate_continuous import get_signals, simple_init, complex_init


def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)


class HMM(object):
	def __init__(self, M, K):
		self.M = M # number of hidden states
		self.K = K # number of Gaussians

	def fit(self, X, max_iter=30, eps=1e-1, debug=False):
		# train the HMM model using the Baum-Welch Algorithm
		# a specific instance of the Expectation-Maximization Algorithm

		# concatenate sequences in X and determine start/end positions
		sequenceLengths = []
		for x in X:
			sequenceLengths.append(len(x))
		Xc = np.concatenate(X)
		T = len(Xc)
		startPositions = np.zeros(T, dtype=np.bool)
		endPositions = np.zeros(T, dtype=np.bool)
		startPositionValues = []
		last = 0
		for length in sequenceLengths:
			startPositionValues.append(last)
			startPositions[last] = 1
			if last > 0:
				endPositions[last - 1] = 1
			last += length

		D = X[0].shape[1] # assume each x is organized (T, D)

		# randomly initialized all parameters
		self.pi = np.ones(self.M) / self.M # initial state distribution
		self.A = random_normalized(self.M, self.M) # state transition matrix
		self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
		self.mu = np.zeros((self.M, self.K, D))
		self.sigma = np.zeros((self.M, self.K, D, D))
		for j in range(self.M):
			for k in range(self.K):
				random_idx = np.random.choice(T)
				self.mu[j,k] = Xc[random_idx]
				self.sigma[j,k] = np.eye(D)

		if debug:
			print('initial pi:\n', self.pi)
			print('initial A:\n', self.A)
			print('initial R:\n', self.R)
			print('initial mu:\n', self.mu)
			print('initial sigma:\n', self.sigma)

		# main EM loop
		costs = []
		for it in range(max_iter):
			# E-step (Expectation Step)
			# calculate B so we can lookup when updating alpha and beta
			B = np.zeros((self.M, T)) # observation probability matrix
			component = np.zeros((self.M, self.K, T)) # we'll need these later
			for j in range(self.M):
				for k in range(self.K):
					p = self.R[j,k] * mvn.pdf(Xc, self.mu[j,k], self.sigma[j,k])
					component[j,k,:] = p
					B[j,:] += p

			scale = np.zeros(T)
			alpha = np.zeros((T, self.M))
			alpha[0] = self.pi * B[:,0] # alpha prime
			scale[0] = alpha[0].sum()
			alpha[0] /= scale[0] # alpha hat
			for t in range(1, T):
				if startPositions[t] == 0:
					alpha_t_prime = alpha[t-1].dot(self.A) * B[:,t]
				else:
					alpha_t_prime = self.pi * B[:,t]
				scale[t] = alpha_t_prime.sum()
				alpha[t] = alpha_t_prime / scale[t]

			if debug:
				logP = np.log(scale).sum()
				costs.append(logP)

			beta = np.zeros((T, self.M))
			beta[-1] = 1
			for t in range(T - 2, -1, -1):
				if startPositions[t + 1] == 1:
					beta[t] = 1
				else:
					beta[t] = self.A.dot(B[:,t+1] * beta[t+1]) / scale[t+1]

			# update for Gaussians
			gamma = np.zeros((T, self.M, self.K))
			for t in range(T):
				alphabeta = alpha[t,:].dot(beta[t,:])
				for j in range(self.M):
					factor = alpha[t,j] * beta[t,j] / alphabeta
					for k in range(self.K):
						gamma[t,j,k] = factor * component[j,k,t] / B[j,t]

			# M-step (Maximization Step)
			self.pi = np.sum((alpha[t] * beta[t]) for t in startPositionValues) / len(startPositionValues)

			# a_den = np.zeros((self.M, 1)) # denominator for A
			a_num = np.zeros((self.M, self.M)) # numerator for A
			r_den = np.zeros(self.M) # denominator for R
			r_num = np.zeros((self.M, self.K)) # numerator for R
			mu_num = np.zeros((self.M, self.K, D)) # numerator for mu
			sigma_num = np.zeros((self.M, self.K, D, D)) # numerator for sigma

			nonEndPositions = (1 - endPositions).astype(np.bool)
			a_den = (alpha[nonEndPositions] * beta[nonEndPositions]).sum(axis=0, keepdims=True).T

			# numeraotr for A
			for i in range(self.M):
				for j in range(self.M):
					for t in range(T-1):
						if endPositions[t] != 1:
							a_num[i,j] += alpha[t,i] * beta[t+1,j] * self.A[i,j] * B[j,t+1] / scale[t+1]
			self.A = a_num / a_den

			# update mixture components
			for j in range(self.M):
				for k in range(self.K):
					for t in range(T):
						r_num[j,k] += gamma[t,j,k]
						r_den[j] += gamma[t,j,k]
						mu_num[j,k] += gamma[t,j,k] * Xc[t]
						diff = Xc[t] - self.mu[j,k]
						sigma_num[j,k] += gamma[t,j,k] * np.outer(diff, diff)

			# update R, mu, sigma
			for j in range(self.M):
				for k in range(self.K):
					self.R[j,k] = r_num[j,k] / r_den[j]
					self.mu[j,k] = mu_num[j,k] / r_num[j,k]
					self.sigma[j,k] = sigma_num[j,k] / r_num[j,k] + np.eye(D) * eps
			assert(np.all(self.R <= 1))
			assert(np.all(self.A <= 1))

		if debug:
			print('updated pi:\n', self.pi)
			print('updated A:\n', self.A)
			print('updated R:\n', self.R)
			print('updated mu:\n', self.mu)
			print('updated sigma:\n', self.sigma)

			plt.plot(costs)
			plt.title('Costs (log-likelihood)')
			plt.show()

	def log_likelihood(self, x):
		# return log P(x | model)
		# using the Forward part of the Forward-Backward Algorithm
		T = len(x)
		B = np.zeros((self.M, T))
		for j in range(self.M):
			for k in range(self.K):
				B[j,:] += self.R[j,k] * mvn.pdf(x, self.mu[j,k], self.sigma[j,k])

		alpha = np.zeros((T, self.M))
		scale = np.zeros(T)
		alpha[0] = self.pi * B[:,0] # alpha prime
		scale[0] = alpha[0].sum()
		alpha[0] /= scale[0] # alpha hat
		for t in range(1, T):
			alpha_t_prime = alpha[t-1].dot(self.A) * B[:,t] # alpha prime
			scale[t] = alpha_t_prime.sum()
			alpha[t] = alpha_t_prime / scale[t] # alpha hat
		return np.log(scale).sum()

	def log_likelihood_multi(self, X):
		return np.array([self.log_likelihood(x) for x in X])

	def set(self, pi, A, R, mu, sigma):
		self.pi = pi
		self.A = A
		self.R = R
		self.mu = mu
		self.sigma = sigma
		M, K = R.shape
		self.M = M
		self.K = K


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
	hmm.fit(signal.reshape(1, T, 1), debug=True)
	LL = hmm.log_likelihood(signal.reshape(T, 1))
	print('LL for fitted params:', LL)


def fake_signal(init=complex_init):
	signals = get_signals(init=init)
	for signal in signals:
		for d in range(signal.shape[1]):
			plt.plot(signal[:,d])
	plt.show()

	hmm = HMM(5, 3)
	hmm.fit(signals, debug=True)
	LL = hmm.log_likelihood_multi(signals).sum()
	print('LL for fitted params:', LL)

	# test in actual params
	_, _, _, pi, A, R, mu, sigma = init()
	hmm.set(pi, A, R, mu, sigma)
	LL = hmm.log_likelihood_multi(signals).sum()
	print('LL for actual params:', LL)


if __name__ == '__main__':
	real_signal()
	# fake_signal(init=simple_init)
	# fake_signal(init=complex_init)

