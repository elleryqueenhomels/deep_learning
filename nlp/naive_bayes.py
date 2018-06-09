# This is an example of a Naive Bayes Classifier on MNIST data.

import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):

	def fit(self, X, Y, smoothing=1e-3):
		# X:(NxD) ; Y:(N, )
		Y = Y.astype(np.int32)

		self.gaussians = dict()
		self.priors = dict()
		
		labels = set(Y)
		for c in labels: # iterate every class in set(Y)
			current_x = X[Y == c]
			self.gaussians[c] = {
				'mean': current_x.mean(axis=0),
				'var': current_x.var(axis=0) + smoothing
			}
			# assert(self.gaussians[c]['mean'].shape[0] == D)
			self.priors[c] = float(len(Y[Y == c])) / len(Y)

	def score(self, X, Y):
		P = self.predict(X)
		Y = Y.astype(np.int32)
		return np.mean(P == Y)

	def predict(self, X):
		N, D = X.shape # X:(NxD) ; N is number of samples, D is number of features.
		K = len(self.priors)
		P = np.zeros((N, K))
		for c, g in self.gaussians.items(): # iterate every class in gaussians or priors or labels
			mean, var = g['mean'], g['var'] # len(mean) == D ; len(var) == D
			# mvn.logpdf can calculate row by row (or say, sample by sample) simultaneously, so can output a (Nx1) vector.
			P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c]) # element-wise
		return np.argmax(P, axis=1)


if __name__ == '__main__':
	from util import get_data

	N = 30000
	X, Y = get_data(N)
	Ntrain = int(N / 2)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	model = NaiveBayes()
	t0 = datetime.now()
	model.fit(Xtrain, Ytrain)
	print('\nTraining time:', (datetime.now() - t0))

	t0 = datetime.now()
	print('\n\nTrain accuracy:', model.score(Xtrain, Ytrain))
	print('Train time:', (datetime.now() - t0), ' Train size:', len(Ytrain))

	t0 = datetime.now()
	print('\n\nTest accuracy:', model.score(Xtest, Ytest))
	print('Test time:', (datetime.now() - t0), ' Test size:', len(Ytest))