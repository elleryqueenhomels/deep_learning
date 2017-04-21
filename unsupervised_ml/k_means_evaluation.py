# Methods to evaluate a clustering, e.g. K-Means and GMM
# external validation method: Purity
# internal validation method: Davies-Bouldin Index

import numpy as np


def purity(T, R):
	# maximum purity is 1, higher is better
	N, K = R.shape
	p = 0
	for k in range(K):
		best_target = -1 # we don't strictly need to store this
		max_intersection = 0
		for j in range(K):
			intersection = R[T==j, k].sum()
			if intersection > max_intersection:
				max_intersection = intersection
				best_target = j
		p += max_intersection
	return p / N


def DBI(X, M, R):
	# Davies-Bouldin Index
	# lower is better
	# N, D = X.shape
	# _, K = R.shape
	K, D = M.shape

	# calculate sigma first
	sigma = np.zeros(K)
	for k in range(K):
		diffs = X - M[k] # should be (N x D)
		# assert(len(diffs.shape) == 2 and diffs.shape[1] == D)
		squared_distances = (diffs * diffs).sum(axis=1)
		# assert(len(squared_distances.shape) == 1 and len(squared_distances) == N)
		weighted_squared_distances = R[:,k] * squared_distances
		sigma[k] = np.sqrt(weighted_squared_distances).mean()

	# calculate Davies-Bouldin Index
	dbi = 0
	for k in range(K):
		max_ratio = 0
		for j in range(K):
			if j != k:
				numerator = sigma[k] + sigma[j]
				denominator = np.linalg.norm(M[k] - M[j]) # By default: Frobenius norm / L2-norm
				ratio = numerator / denominator
				if ratio > max_ratio:
					max_ratio = ratio
		dbi += max_ratio
	return dbi / K

