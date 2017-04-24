import numpy as np
import pandas as pd

DATA_PATH = '../../python_test/data_set/MNIST_train.csv'


def error_rate(T, P):
	return np.mean(T != P)

def classification_rate(T, P):
	return np.mean(T == P)

def init_weights(shape):
	W = np.random.randn(*shape) / np.sqrt(sum(shape))
	return W.astype(np.float32)

def get_mnist(Ntrain=-1000):
	# MNIST data:
	# column 0 is labels
	# column 1-785 is data, with values 0..255
	# total size of csv: (42000, 1, 28, 28)
	train = pd.read_csv(DATA_PATH).as_matrix().astype(np.float32)
	np.random.shuffle(train)

	Xtrain = train[:Ntrain, 1:] / np.float32(255)
	Ytrain = train[:Ntrain, 0].astype(np.int32)

	Xtest = train[Ntrain:, 1:] / np.float32(255)
	Ytest = train[Ntrain:, 0].astype(np.int32)

	return Xtrain, Ytrain, Xtest, Ytest

