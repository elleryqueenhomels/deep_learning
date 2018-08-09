# Decision Tree for continuous-vector input, binary output(0 or 1)
import numpy as np
from datetime import datetime
from util import get_data, get_xor, get_donut


def entropy(y):
	# assumes that y is binary -- 0 or 1
	N = len(y)
	s1 = (y == 1).sum()
	if s1 == 0 or s1 == N:
		return 0
	p1 = float(s1) / N
	p0 = 1 - p1
	return -p0*np.log2(p0) - p1*np.log2(p1)


class TreeNode:
	def __init__(self, depth=0, max_depth=None):
		self.depth = depth
		self.max_depth = max_depth

	def fit(self, X, Y):
		if len(Y) == 1 or len(set(Y)) == 1:
			# base case, only 1 sample
			# another base case
			# this node only receives samples from one class
			# we cannot make a split
			self.col = None
			self.split = None
			self.left = None
			self.right = None
			self.prediction = Y[0]
		else:
			D = X.shape[1]
			cols = range(D)

			max_ig = 0 # ig is short for information gain
			best_col = None
			best_split = None
			for col in cols:
				ig, split = self.find_split(X, Y, col)
				if ig > max_ig:
					max_ig = ig
					best_col = col
					best_split = split

			if max_ig == 0:
				# nothing we can do
				# no further splits
				self.col = None
				self.split = None
				self.left = None
				self.right = None
				self.prediction = np.round(Y.mean())
			else:
				self.col = best_col
				self.split = best_split

				if self.depth == self.max_depth:
					self.left = None
					self.right = None
					self.prediction = [
						np.round(Y[X[:, best_col] < best_split].mean()),
						np.round(Y[X[:, best_col] >= best_split].mean())
					]
				else:
					left_idx = (X[:, best_col] < best_split)
					Xleft = X[left_idx]
					Yleft = Y[left_idx]
					self.left = TreeNode(self.depth + 1, self.max_depth)
					self.left.fit(Xleft, Yleft)

					right_idx = (X[:, best_col] >= best_split)
					Xright = X[right_idx]
					Yright = Y[right_idx]
					self.right = TreeNode(self.depth + 1, self.max_depth)
					self.right.fit(Xright, Yright)

	def find_split(self, X, Y, col):
		x_values = X[:, col]
		sort_idx = np.argsort(x_values)
		x_values = x_values[sort_idx]
		y_values = Y[sort_idx]

		# Note: optimal split is the midpoint between 2 points
		# Note: optimal split is only on the boundaries between 2 classes (0 -> 1 or 1 -> 0)

		# if boundaries[i] is true
		# then y_values[i] != y_values[i+1]
		# nonzero() gives us indices where arg is true
		# but for some reason it returns a tuple of size 1
		boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
		max_ig = 0
		best_split = None
		for b in boundaries:
			split = (x_values[b] + x_values[b+1]) / 2
			ig = self.information_gain(x_values, y_values, split)
			if ig > max_ig:
				max_ig = ig
				best_split = split
		return max_ig, best_split

	def information_gain(self, x, y, split):
		# assumes classes are 0 and 1
		y0 = y[x < split]
		y1 = y[x >= split]
		N = len(y)
		y0len = len(y0)
		if y0len == 0 or y0len == N:
			return 0
		p0 = float(len(y0)) / N
		p1 = 1 - p0
		return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

	def predict_one(self, x):
		# x is just only one row of input matrix X
		# use 'is not None' because 0 means False
		if self.col is not None and self.split is not None:
			feature = x[self.col]
			if feature < self.split:
				if self.left:
					p = self.left.predict_one(x)
				else:
					p = self.prediction[0]
			else:
				if self.right:
					p = self.right.predict_one(x)
				else:
					p = self.prediction[1]
		else:
			# corresponds to having only 1 prediction
			p = self.prediction
		return p

	def predict(self, X):
		N = len(X)
		P = np.zeros(N)
		for i in range(N):
			P[i] = self.predict_one(X[i])
		return P


# This class is kind of redundant
class DecisionTree:
	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	def fit(self, X, Y):
		self.root = TreeNode(max_depth=self.max_depth)
		self.root.fit(X, Y)

	def predict(self, X):
		return self.root.predict(X)

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)


if __name__ == '__main__':
	X, Y = get_data()

	# try donut or xor
	# from sklearn.utils import shuffle
	# X, Y = get_xor()
	# X, Y = get_donut()
	# X, Y = shuffle(X, Y)

	# only takes 0s and 1s since we are doing binary classification
	idx = np.logical_or(Y == 0, Y == 1)
	X = X[idx]
	Y = Y[idx]

	# split the data
	Ntrain = int(len(Y) / 2)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# model = DecisionTree()
	model = DecisionTree(max_depth=7)

	t0 = datetime.now()
	model.fit(Xtrain, Ytrain)
	print('\nTraining time:', (datetime.now() - t0))

	t0 = datetime.now()
	print('\nTrain accuracy:', model.score(Xtrain, Ytrain))
	print('Train accuracy time:', (datetime.now() - t0), ' Train size:', len(Ytrain))

	t0 = datetime.now()
	print('\nTest accuracy:', model.score(Xtest, Ytest))
	print('Test accuracy time:', (datetime.now() - t0), ' Test size:', len(Ytest))

