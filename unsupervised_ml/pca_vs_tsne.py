# A demo to compare PCA and t-SNE
# Using GMM to do clustering analysis
# Using Purity to measure the "accuracy"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import get_mnist
from k_means_evaluation import purity
from gmm import gmm


def main():
	Xtrain, Ytrain, _, _ = get_mnist()

	sample_size = 1000 # if sample_size is too large and your computer has not enough RAM, t-SNE may crash.
	X = Xtrain[:sample_size]
	Y = Ytrain[:sample_size]
	K = len(set(Y))
	print('K =', K)

	# PCA
	pca = PCA()
	Z_pca = pca.fit_transform(X)
	print('\nZ_pca.shape:', Z_pca.shape)
	plt.scatter(Z_pca[:,0], Z_pca[:,1], c=Y, s=100, alpha=0.5)
	plt.title('PCA')
	plt.show()

	R_pca, _, _, _ = gmm(Z_pca[:, :2], K, eps=0)
	print('PCA purity:', purity(Y, R_pca))

	# t-SNE
	tsne = TSNE()
	Z_tsne = tsne.fit_transform(X)
	print('\nZ_tsne.shape:', Z_tsne.shape)
	plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=Y, s=100, alpha=0.5)
	plt.title('t-SNE')
	plt.show()

	R_tsne, _, _, _ = gmm(Z_tsne, K, eps=0)
	print('t-SNE purity:', purity(Y, R_tsne))


if __name__ == '__main__':
	main()

