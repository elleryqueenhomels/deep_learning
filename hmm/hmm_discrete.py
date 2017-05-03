# Discrete Hidden Markov Model

import numpy as np
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)


class HMM(object):
	def __init__(self, M):
		self.M = M # number of hidden states

	def fit(self, X, max_iter=30, seed=123, debug=False):
		np.random.seed(seed)
		# train the HMM model using the Baum-Welch Algorithm
		# a specific instance of the Expectation-Maximization Algorithm

		# determine V, the vocabulary size (the observation symbol size)
		# assume observables are already integers from 0..V-1
		# X is a jagged array of observed sequences
		V = max(max(x) for x in X) + 1
		N = len(X)

		self.pi = np.ones(self.M) / self.M # initial state distribution
		self.A = random_normalized(self.M, self.M) # state transition matrix
		self.B = random_normalized(self.M, V) # observation distribution

		if debug:
			print('initial A:\n', self.A)
			print('initial B:\n', self.B)
			print('initial pi:\n', self.pi)

		# update pi, A, B
		costs = []
		for it in range(max_iter):
			# E-step (Expectation Step)
			alphas = []
			betas = []
			P = np.zeros(N)
			for n in range(N):
				x = X[n]
				T = len(x)

				alpha = np.zeros((T, self.M))
				alpha[0] = self.pi * self.B[:, x[0]]
				for t in range(1, T):
					alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
				P[n] = alpha[-1].sum()
				alphas.append(alpha)

				beta = np.zeros((T, self.M))
				beta[-1] = 1
				for t in range(T - 2, -1, -1):
					beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])
				betas.append(beta)

			if debug:
				cost = np.sum(np.log(P))
				costs.append(cost)

			# M-step (Maximization Step)
			self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N)) / N

			den1 = np.zeros((self.M, 1)) # denominator for A
			den2 = np.zeros((self.M, 1)) # denominator for B
			a_num = np.zeros((self.M, self.M)) # numerator for A
			b_num = np.zeros((self.M, V)) # numerator for B
			for n in range(N):
				x = X[n]
				T = len(x)

				den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
				den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

				# numerator for A
				a_num_n = np.zeros((self.M, self.M))
				for i in range(self.M):
					for j in range(self.M):
						for t in range(T-1):
							a_num_n[i, j] += alphas[n][t,i] * self.A[i,j] * self.B[j, x[t]] * betas[n][t+1,j]
				a_num += a_num_n / P[n]

				# numerator for B
				b_num_n = np.zeros((self.M, V))
				for j in range(self.M):
					for t in range(T):
						b_num_n[j, x[t]] += alphas[n][t,j] * betas[n][t,j]
				b_num += b_num_n / P[n]

			self.A = a_num / den1
			self.B = b_num / den2

		if debug:
			print('updated A:\n', self.A)
			print('updated B:\n', self.B)
			print('updated pi:\n', self.pi)

			plt.plot(costs)
			plt.title('Costs (log-likelihood)')
			plt.show()

	def likelihood(self, x):
		# returns P(x | model) -- the likelihood of the sequence of observations
		# using the forward part of the Forward-Backward Algorithm
		T = len(x)
		alpha = np.zeros((T, self.M))
		alpha[0] = self.pi * self.B[:,x[0]]
		for t in range(1, T):
			alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
		return alpha[-1].sum()

	def likelihood_multi(self, X):
		return np.array([self.likelihood(x) for x in X])

	def log_likelihood_multi(self, X):
		return np.log(self.likelihood_multi(X))

	def get_state_sequence(self, x):
		# returns the most-likely hidden state sequence given observed sequence x
		# using the Viterbi Algorithm
		T = len(x)
		delta = np.zeros((T, self.M))
		psi = np.zeros((T, self.M))
		delta[0] = self.pi * self.B[:,x[0]]
		for t in range(1, T):
			for j in range(self.M):
				delta[t,j] = np.max(delta[t-1] * self.A[:,j]) * self.B[j, x[t]]
				psi[t,j] = np.argmax(delta[t-1] * self.A[:,j])

		# backtracking
		p = np.max(delta[-1]) # if necessary, we can return this p -- probability of the hidden state sequence
		states = np.zeros(T, dtype=np.int32)
		states[-1] = np.argmax(delta[-1])
		for t in range(T - 2, -1, -1):
			states[t] = psi[t+1, states[t+1]]
		return states


# demo
def fit_coin():
	X = []
	for line in open('data_set/coin_data.txt'):
		# 1 for H, 0 for T
		x = [1 if e == 'H' else 0 for e in line.rstrip()]
		X.append(x)

	hmm = HMM(2)
	hmm.fit(X, max_iter=30, seed=123, debug=True)
	L = hmm.log_likelihood_multi(X).sum()
	print('LL with fitted params:', L)

	# try true values
	hmm.pi = np.array([0.5, 0.5])
	hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
	hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
	L = hmm.log_likelihood_multi(X).sum()
	print('LL with true params:', L)

	# try Viterbi
	Y = np.array(X[0])
	P = hmm.get_state_sequence(X[0])
	print('True hidden state sequence:\n', Y)
	print('Viterbi hidden state sequence:\n', P)
	print('Accuracy: %.6f%%' % (np.mean(Y == P)*100))


if __name__ == '__main__':
	fit_coin()

