# A demo to compare Soft K-Means and GMM
# Using Purity to measure the "accuracy"

import numpy as np

from sklearn.manifold import TSNE
from util import get_mnist
from k_means_evaluation import purity
from k_means import plot_k_means
from gmm import gmm


def main():
	_, _, X, Y = get_mnist()

	K = len(set(Y))
	print('K =', K)

	print('\nBefore dimensionality reduction:')
	print('input data shape:', X.shape)

	_, R_kmeans = plot_k_means(X, K, max_iter=30)
	print('K-Means purity:', purity(Y, R_kmeans))

	R_gmm, _, _, _ = gmm(X, K, max_iter=30, smoothing=1, eps=1e-16)
	print('GMM purity:', purity(Y, R_gmm))

	print('\nAfter dimensionality reduction with t-SNE:')

	tsne = TSNE()
	Z = tsne.fit_transform(X)
	print('input data shape:', Z.shape)

	_, R_kmeans_tsne = plot_k_means(Z, K, max_iter=30)
	print('K-Means purity:', purity(Y, R_kmeans_tsne))

	R_gmm_tsne, _, _, _ = gmm(Z, K, max_iter=30, eps=0)
	print('GMM purity:', purity(Y, R_gmm_tsne))


if __name__ == '__main__':
	main()

