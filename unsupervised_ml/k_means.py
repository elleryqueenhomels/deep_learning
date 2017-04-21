# Clustering Analysis: Soft K-Means
import numpy as np
import matplotlib.pyplot as plt


def d(u, v):
	diff = u - v
	return diff.dot(diff)


def cost(X, R, M):
	cost = 0
	# method 1
	# for k in range(len(M)):
	# 	for n in range(len(X)):
	# 		cost += R[n,k] * d(M[k], X[n])

	# method 2
	for k in range(len(M)):
		diff = X - M[k] # broadcasting M[k] via each row
		dist_sq = (diff * diff).sum(axis=1)
		cost += R[:,k].dot(dist_sq)

	return cost


def plot_k_means(X, K, max_iter=100, beta=1.0, show_plots=True):
	N, D = X.shape
	M = np.zeros((K, D)) # K Means / K Clusters
	# R = np.zeros((N, K)) # responsibilities matrix
	exponents = np.zeros((N, K))

	# first: initialize means to random points in X
	for k in range(K):
		M[k] = X[np.random.choice(N)]

	# second: K-Means until converged
	costs = np.zeros(max_iter)
	for i in range(max_iter):
		# step 1: calculate cluster resposibilities.
		# (determine assignments / responsibilities)
		for n in range(N):
			for k in range(K):
				# R[n, k] = np.exp(-beta*d(M[k], X[n])) / np.sum( np.exp(-beta*d(M[j], X[n])) for j in range(K) )
				exponents[n, k] = np.exp(-beta*d(M[k], X[n]))
		R = exponents / exponents.sum(axis=1, keepdims=True) # just similar to softmax

		# step 2: recalculate means.
		# for k in range(K):
		# 	M[k] = R[:,k].dot(X) / R[:,k].sum()
		M = R.T.dot(X) / R.sum(axis=0, keepdims=True).T

		costs[i] = cost(X, R, M)
		if i > 0:
			if np.abs(costs[i] - costs[i-1]) < 10e-7:
				break

	if show_plots:
		plt.plot(costs)
		plt.title('Costs')
		plt.show()

		random_colors = np.random.random((K, 3))
		colors = R.dot(random_colors)
		plt.scatter(X[:,0], X[:,1], c=colors, s=100, alpha=0.5)
		plt.show()

	return M, R


def get_simple_data():
	# assume 3 means
	D = 2 # so we can visualize it more easily
	s = 4 # separation so we can control how far apart the means are
	mu1 = np.array([0, 0])
	mu2 = np.array([s, s])
	mu3 = np.array([0, s])

	N = 900 # number of samples
	X = np.zeros((N, D))
	X[:300, :] = np.random.randn(300, D) + mu1
	X[300:600, :] = np.random.randn(300, D) + mu2
	X[600:, :] = np.random.randn(300, D) + mu3

	return X


def main():
	X = get_simple_data()

	# what does it look like without clustering?
	plt.scatter(X[:,0], X[:,1])
	plt.show()

	K = 3 # luckily, we already know this
	plot_k_means(X, K)

	K = 5 # what happens if we choose a "bad" K?
	plot_k_means(X, K, max_iter=30)

	K = 5 # what happens if we change beta?
	plot_k_means(X, K, max_iter=30, beta=0.3)


if __name__ == '__main__':
	main()
