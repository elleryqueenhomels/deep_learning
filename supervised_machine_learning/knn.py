# This is an example of a K-Nearest Neighbors classifier on MNIST Data.
# We try K=1..5 to show how we might choose the best K.

import numpy as np
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime


class KNN(object):
	def __init__(self, K):
		self.K = K

	def fit(self, X, Y):
		self.X = X # X: (NxD)
		self.Y = Y # Y: (Nx1)

	# This is training.
	def predict(self, X):    # X: (NxD)
		Y = np.zeros(len(X)) # Y: (Nx1)
		for i, x in enumerate(X): # iterate every sample/row in input X
			sl = SortedList(load=self.K) # stores K (distance, class) tuples
			for j, xt in enumerate(self.X): # training points
				diff = x - xt
				dist = diff.dot(diff)
				if len(sl) < self.K:
					sl.add((dist, self.Y[j]))
				else:
					if dist < sl[-1][0]:
						del sl[-1]
						sl.add((dist, self.Y[j]))

			votes = {} # {} means a dictionary.
			for _, v in sl:
				votes[v] = votes.get(v, 0) + 1
			max_votes = 0
			max_votes_class = -1
			for v, count in votes.items():
				if count > max_votes:
					max_votes = count
					max_votes_class = v
			Y[i] = max_votes_class
		return Y

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)


if __name__ == '__main__':
	X, Y = get_data(2000)
	Ntrain = 1000
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
	for K in (1, 2, 3, 4, 5):
		knn = KNN(K)

		print('\n\n\nThe %dth iteration' % K)

		t0 = datetime.now()
		knn.fit(Xtrain, Ytrain)
		print('Training time:', (datetime.now() - t0))

		t0 = datetime.now()
		print('\nTrain accuracy:', knn.score(Xtrain, Ytrain))
		print('Time to compute train accuracy:', (datetime.now() - t0), ' Train size:', len(Ytrain))

		t0 = datetime.now()
		print('\nTest accuracy:', knn.score(Xtest, Ytest))
		print('Time to compute test accuracy:', (datetime.now() - t0), ' Test size:', len(Ytest))