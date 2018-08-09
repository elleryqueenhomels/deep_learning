import numpy as np
from ann import ANN
from util import getBinaryData, getData
from sklearn.utils import shuffle


def test_binary():
	X, Y = getBinaryData()

	# balance ones
	X0 = X[Y == 0]
	X1 = X[Y == 1]
	X1 = np.repeat(X1, 9, axis=0)
	X = np.vstack([X0, X1])
	Y = np.array([0]*len(X0) + [1]*len(X1))

	X, Y = shuffle(X, Y)

	Ntrain = int(len(Y) / 2)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	print('\nTrain size:', len(Ytrain), '\n')

	model = ANN([100], activation_type=2)
	model.fit(Xtrain, Ytrain, learning_rate=10e-7, epochs=10000)
	score = model.score(Xtest, Ytest)

	print('\nFinal Score: %.8f%%' % (score * 100), ' Test size:', len(Ytest), '\n')


def test_all():
	X, Y = getData()
	X, Y = shuffle(X, Y)

	Ntrain = int(len(Y) / 2)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	print('\nTrain size:', len(Ytrain), '\n')

	model = ANN([100], activation_type=2)
	model.fit(Xtrain, Ytrain, learning_rate=10e-7, epochs=10000)
	score = model.score(Xtest, Ytest)

	print('\nFinal Score: %.8f%%' % (score * 100), ' Test size:', len(Ytest), '\n')


if __name__ == '__main__':
	test_all()
