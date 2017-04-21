# A demo to compare Soft K-Means and GMM
# Using Purity to measure the "accuracy"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from util import get_mnist
from k_means_evaluation import purity
from k_means import plot_k_means
from gmm import gmm


def main():
	_, _, X, Y = get_mnist()

	K = len(set(Y))
	max_iter = 50
	print('K =', K)

	print('\nBefore dimensionality reduction:')
	print('input data shape:', X.shape)

	_, R_kmeans = plot_k_means(X, K, max_iter=max_iter)
	print('K-Means purity:', purity(Y, R_kmeans))

	R_gmm, _, _, _ = gmm(X, K, max_iter=max_iter, smoothing=1, eps=1e-16)
	print('GMM purity:', purity(Y, R_gmm))

	print('\nAfter dimensionality reduction with t-SNE:')

	tsne = TSNE()
	Z = tsne.fit_transform(X)
	print('input data shape:', Z.shape)
	
	plt.scatter(Z[:,0], Z[:,1], c=Y, s=100, alpha=0.5)
	plt.title('Original Distribution with True Labels before Clustering')
	plt.show()
	
	_, R_kmeans_tsne = plot_k_means(Z, K, max_iter=max_iter)
	print('K-Means purity:', purity(Y, R_kmeans_tsne))

	R_gmm_tsne, _, _, _ = gmm(Z, K, max_iter=max_iter, eps=0)
	print('GMM purity:', purity(Y, R_gmm_tsne))


if __name__ == '__main__':
	main()

