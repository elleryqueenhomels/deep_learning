# Gaussian Mixture Model
# K-Means is a special case of GMM

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn


def gmm(X, K, max_iter=100, smoothing=10e-3, eps=1e-12, show_plots=True):
	N, D = X.shape
	M = np.zeros((K, D))
	R = np.zeros((N, K))
	C = np.zeros((K, D, D))
	pi = np.ones(K) / K # initialize to uniform distribution

	# initialize M to random points in X, initialize C to spherical with variance 1
	for k in range(K):
		M[k] = X[np.random.choice(N)]
		C[k] = np.eye(D)

	costs = np.zeros(max_iter)
	weighted_pdfs = np.zeros((N, K)) # we'll use these to store the PDF value of sample n and Gaussian k
	for i in range(max_iter):
		# step 1: determine assignments / responsibilities
		# a naive method:
		# for k in range(K):
		# 	for n in range(N):
		# 		weighted_pdfs[n,k] = pi[k] * mvn.pdf(X[n], M[k], C[k])

		# for k in range(K):
		# 	for n in range(N):
		# 		R[n,k] = weighted_pdfs[n,k] / weighted_pdfs[n,:].sum()

		# a faster method to do step 1: vectorize
		for k in range(K):
			# weighted_pdfs[:,k] = pi[k] * mvn.pdf(X, M[k], C[k])
			weighted_pdfs[:,k] = pi[k] * mvn.pdf(X, M[k], C[k]) + eps # adding an eps to avoid any row being all zeros

		R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)


		# step 2: recalculate model parameters
		# a naive method
		# for k in range(K):
		# 	Nk = R[:,k].sum()
		# 	pi[k] = Nk / N
		# 	M[k] = R[:,k].dot(X) / Nk
		# 	C[k] = np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in range(N)) / Nk + np.eye(D)*smoothing

		# a faster method to do step 2: vectorize
		Nk = R.sum(axis=0)
		pi = Nk / N
		M = R.T.dot(X) / Nk.reshape(K, 1)
		for k in range(K):
			X_minus_Mean = X - M[k]
			cov = (R[:,k] * X_minus_Mean.T).dot(X_minus_Mean) / Nk[k]
			C[k] = cov + np.eye(D)*smoothing


		costs[i] = -np.log(weighted_pdfs.sum(axis=1)).sum()
		if i > 0:
			if np.abs(costs[i] - costs[i-1]) < 10e-7:
				print('\nEarly break at %d iterations\n' % (i+1))
				break

	if show_plots:
		plt.plot(costs)
		plt.title('Costs with %d iterations' % (i+1))
		plt.show()

		random_colors = np.random.random((K, 3))
		colors = R.dot(random_colors)
		plt.scatter(X[:,0], X[:,1], c=colors)
		plt.show()

		print('pi:', pi)
		print('means:', M)
		print('covariances:', C)

	return R, pi, M, C


def main():
	# assume 3 means
	D = 2 # so we can visualize it more easily
	s = 6 # separation so we can control how far apart the means are
	mu1 = np.array([0, 0])
	mu2 = np.array([s, s])
	mu3 = np.array([0, s])

	N = 2000 # number of samples
	X = np.zeros((N, D))
	X[:1200, :] = np.random.randn(1200, D)*2 + mu1
	X[1200:1800, :] = np.random.randn(600, D) + mu2
	X[1800:, :] = np.random.randn(200, D)*0.5 + mu3

	# what does it look like without clustering?
	plt.scatter(X[:,0], X[:,1])
	plt.show()

	K = 3
	gmm(X, K)


if __name__ == '__main__':
	main()

