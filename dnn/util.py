# Utility for: AutoEncoder, RBM, DNN

import numpy as np
import pandas as pd
import theano.tensor as T


def init_filter(shape, poolsz):
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return W.astype(np.float32)


def init_weight_and_bias(M1, M2):
	M1, M2 = int(M1), int(M2)
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def init_weights(shape):
	W = np.random.randn(*shape) / np.sqrt(sum(shape))
	return W.astype(np.float32)


def get_activation(activation_type):
	if activation_type == 1:
		return T.nnet.sigmoid
	elif activation_type == 2:
		return T.tanh
	elif activation_type == 3:
		return T.nnet.relu
	else:
		return elu


def elu(X):
	return T.switch(X >= np.float32(0), X, (T.exp(X) - np.float32(1)))


def preprocess(X, Y, debug):
	if debug:
		if X is None or Y is None or len(X) != len(Y):
			return None, None, False
	if len(X.shape) == 1:
		X = X.reshape(-1, 1)
	if len(Y.shape) == 2:
		if Y.shape[1] == 1:
			Y = np.squeeze(Y)
		else:
			Y = np.argmax(Y, axis=1)
	X = X.astype(np.float32)
	Y = Y.astype(np.int32)
	return X, Y, debug


def preprocess_cnn(X, Y, debug):
	if debug:
		if X is None or Y is None or len(X) != len(Y):
			return None, None, False
	if len(Y.shape) == 2:
		if Y.shape[1] == 1:
			Y = np.squeeze(Y)
		else:
			Y = np.argmax(Y, axis=1)
	X = X.astype(np.float32)
	Y = Y.astype(np.int32)
	return X, Y, debug


def error_rate(T, P):
	return np.mean(T != P)


def classification_rate(T, P):
	return np.mean(T == P)


def shuffle(X, Y=None):
	if Y is None:
		np.random.shuffle(X)
		return X

	idx = np.arange(len(X))
	np.random.shuffle(idx)
	return X[idx], Y[idx]


def get_mnist(Ntrain=-1000):
	# MNIST data:
	# column 0 is labels
	# column 1-785 is data, with values 0..255
	# total size of csv: (42000, 1, 28, 28)
	DATA_PATH = '../../python_test/data_set/MNIST_train.csv'
	train = pd.read_csv(DATA_PATH).as_matrix().astype(np.float32)
	np.random.shuffle(train)

	Xtrain = train[:Ntrain, 1:] / np.float32(255)
	Ytrain = train[:Ntrain, 0].astype(np.int32)

	Xtest = train[Ntrain:, 1:] / np.float32(255)
	Ytest = train[Ntrain:, 0].astype(np.int32)

	return Xtrain, Ytrain, Xtest, Ytest

