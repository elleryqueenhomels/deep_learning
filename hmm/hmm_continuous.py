# Continuous-observation HMM with no scaling, but allowing multiple observations

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

		N = len(X)
		D = X[0].shape[1] # assume each x is organized (T, D)

		self.pi = np.ones(self.M) / self.M # initial state distribution
		self.A = random_normalized(self.M, self.M) # state transition matrix
		self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
		self.mu = np.zeros((self.M, self.K, D))
		self.sigma = np.zeros((self.M, self.K, D, D))
		for j in range(self.M):
			for k in range(self.K):
				random_idx = np.random.choice(N)
				x = X[random_idx]
				random_time_idx = np.random.choice(len(x))
				self.mu[j,k] = x[random_time_idx]
				self.sigma[j,k] = np.eye(D)

		if debug:
			print('initial pi:\n', self.pi)
			print('initial A:\n', self.A)
			print('initial R:\n', self.R)

		costs = []
		for it in range(max_iter):
			# E-step (Expectation Step)
			alphas = []
			betas = []
			gammas = []
			Bs = []
			P = np.zeros(N)

			for n in range(N):
				x = X[n]
				T = len(x)

				# calculate B so we can lookup when updating alpha and beta
				B = np.zeros((self.M, T))
				component = np.zeros((self.M, self.K, T)) # we will need these later
				for j in range(self.M):
					for t in range(T):
						for k in range(self.K):
							p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
							component[j,k,t] = p
							B[j,t] += p
				Bs.append(B)

				alpha = np.zeros((T, self.M))
				alpha[0] = self.pi * B[:,0]
				for t in range(1, T):
					alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
				P[n] = alpha[-1].sum()
				alphas.append(alpha)
				assert(P[n] <= 1)

				beta = np.zeros((T, self.M))
				beta[-1] = 1
				for t in range(T - 2, -1, -1):
					beta[t] = self.A.dot(B[:,t+1] * beta[t+1])
				betas.append(beta)

				# update for Gaussians
				gamma = np.zeros((T, self.M, self.K))
				for t in range(T):
					alphabeta = (alpha[t,:] * beta[t,:]).sum()
					for j in range(self.M):
						factor = alpha[t,j] * beta[t,j] / alphabeta
						for k in range(self.K):
							gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
				gammas.append(gamma)

			if debug:
				cost = -np.log(P).sum()
				costs.append(cost)

			# M-step (Maximization Step)
			self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N)) / N

			a_den = np.zeros((self.M, 1)) # denominator for A
			a_num = np.zeros((self.M, self.M)) # numerator for A
			r_den = np.zeros(self.M) # denominator for R
			r_num = np.zeros((self.M, self.K)) # numerator for R, also used for denominator for mu and sigma
			mu_num = np.zeros((self.M, self.K, D)) # numerator for mu
			sigma_num = np.zeros((self.M, self.K, D, D)) # numerator for sigma
			for n in range(N):
				x = X[n]
				T = len(x)
				B = Bs[n]
				gamma = gammas[n]

				# denominator for A
				a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

				# numerator for A
				a_num_n = np.zeros((self.M, self.M))
				for i in range(self.M):
					for j in range(self.M):
						for t in range(T-1):
							a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
				a_num += a_num_n / P[n]

				# update mixture components
				r_num_n = np.zeros((self.M, self.K))
				r_den_n = np.zeros(self.M)
				for j in range(self.M):
					for k in range(self.K):
						for t in range(T):
							r_num_n[j,k] += gamma[t,j,k]
							r_den_n[j] += gamma[t,j,k]
				r_num += r_num_n / P[n]
				r_den += r_den_n / P[n]

				mu_num_n = np.zeros((self.M, self.K, D))
				sigma_num_n = np.zeros((self.M, self.K, D, D))
				for j in range(self.M):
					for k in range(self.K):
						for t in range(T):
							mu_num_n[j,k] += gamma[t,j,k] * x[t]
							diff = x[t] - self.mu[j,k]
							sigma_num_n[j,k] += gamma[t,j,k] * np.outer(diff, diff)
				mu_num += mu_num_n / P[n]
				sigma_num += sigma_num_n / P[n]

			self.A = a_num / a_den
			assert(np.all(self.A <= 1))

			for j in range(self.M):
				for k in range(self.K):
					self.R[j,k] = r_num[j,k] / r_den[j]
					self.mu[j,k] = mu_num[j,k] / r_num[j,k]
					self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

		if debug:
			print('updated pi:\n', self.pi)
			print('updated A:\n', self.A)
			print('updated R:\n', self.R)
			print('updated mu:\n', self.mu)
			print('updated sigma:\n', self.sigma)

			plt.plot(costs)
			plt.title('Costs (log-likelihood)')
			plt.show()

	def likelihood(self, x):
		# returns log P(x | model)
		# using the Forward part of the Forward-Backward Algorithm
		T = len(x)
		alpha = np.zeros((T, self.M))

		B = np.zeros((self.M, T))
		for j in range(self.M):
			for t in range(T):
				for k in range(self.K):
					B[j,t] += self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])

		alpha[0] = self.pi * B[:,0]
		for t in range(1, T):
			alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
		return alpha[-1].sum()

	def likelihood_multi(self, X):
		return np.array([self.likelihood(x) for x in X])

	def log_likelihood_multi(self, X):
		return np.log(self.likelihood_multi(X))

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

	hmm = HMM(5,3)
	hmm.fit(signal.reshape(1, T), debug=True)


def fake_signal(init=simple_init):
	signals = get_signals(N=10, T=10, init=init)
	for signal in signals:
		for d in range(signal.shape[1]):
			plt.plot(signal[:,d])
	plt.show()

	hmm = HMM(2, 2)
	hmm.fit(signals, debug=True)
	print('\nAfter training...\n')

	L = hmm.log_likelihood_multi(signals).sum()
	print('LL for fitted params:', L)

	# test in actual params
	_, _, _, pi, A, R, mu, sigma = init()
	hmm.set(pi, A, R, mu, sigma)
	L = hmm.log_likelihood_multi(signals).sum()
	print('LL for actual params:', L)


if __name__ == '__main__':
	# real_signal() # will break
	fake_signal(init=simple_init)
	# fake_signal(init=complex_init) # will break

