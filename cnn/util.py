import numpy as np
import theano.tensor as T


def init_filter(shape, poolsz):
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return W.astype(np.float32)


def init_weight_and_bias(M1, M2):
	M1, M2 = int(M1), int(M2)
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


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


def classification_rate(T, P):
	return np.mean(T == P)


def error_rate(T, P):
	return np.mean(T != P)


def shuffle(X, Y):
	idx = np.arange(len(Y))
	np.random.shuffle(idx)
	return X[idx], Y[idx]


def rearrange(X):
	# input is (W, H, C, N) from matlab file
	# output is (N, C, W, H) for Theano using
	W, H, C, N = X.shape
	out = np.zeros((N, C, W, H), dtype=np.float32)
	for i in range(N):
		for j in range(C):
			out[i, j, :, :] = X[:, :, j, i]
	return out / np.float32(255)
