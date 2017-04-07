# Vanilla deep network

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime


def y2indicator(Y, K=None):
	N = len(Y)
	if K == None:
		K = len(set(Y))
	T = np.zeros((N, K))
	T[np.arange(N), Y.astype(np.int32)] = 1
	return T


def error_rate(T, P):
	return np.mean(T != P)


def flatten(X):
	# input will be (32, 32, 3, N)
	# output will be (N, 3072)
	N = X.shape[-1]
	K = 3072
	flat = np.zeros((N, K))
	for i in range(N):
		flat[i] = X[:,:,:,i].reshape(K)
	return flat


def main():
	train = loadmat('../data_set/train_32x32.mat')
	test = loadmat('../data_set/test_32x32.mat')

	# Need to scale! Don't leave as 0..255
	# Y is a (N x 1) matrix with values 1..10 (MATLAB indexes by 1)
	# So flatten it and make it 0..9
	# Also need indicator matrix for cost calculation
	Xtrain = flatten(train['X'].astype(np.float32) / 255)
	Ytrain = train['y'].flatten() - 1
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Ytrain_ind = y2indicator(Ytrain, 10)

	Xtest = flatten(test['X'].astype(np.float32) / 255)
	Ytest = test['y'].flatten() - 1
	Ytets_ind = y2indicator(Ytest, 10)

	# gradient descent params
	max_iter = 20
	print_period = 10
	N, D = Xtrain.shape
	batch_sz = 500
	n_batches = int(N / batch_sz)
	lr = 10e-5
	reg = 0.01
	decay = 0.99
	mu = 0.9

	# initialize weights
	M1 = 1000 # hidden layer size
	M2 = 500
	K = 10
	W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
	b1_init = np.zeros(M1)
	W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b2_init = np.zeros(M2)
	W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
	b3_init = np.zeros(K)

	# define variables and expressions
	tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
	tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Varaible(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Varaible(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Varaible(b3_init.astype(np.float32))

	Z1 = tf.nn.relu(tf.matmul(Xtrain, W1) + b1)
	Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
	Yish = tf.matmul(Z2, W3) + b3

	reg_cost = reg * sum([tf.nn.l2_loss(p) for p in [W1, b1, W2, b2, W3, b3]])
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=tfT)) + reg_cost

	train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

	# we will use this to calculate the error rate
	predict_op = tf.argmax(Yish, 1)

	t0 = datetime.now()
	costs = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			Xtrain, Ytrain = shuffle(Xtrain, Ytrain_ind)
			for j in range(n_batches):
				Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
				Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz)]

				session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={tfX: Xtest, tfT: Ytets_ind})
					test_predict = session.run(predict_op, feed_dict={tfX: Xtest})
					err = error_rate(Ytest, test_predict)
					print('Cost / error_rate at iteration i=%d, j=%d: %.3f / %.3f%%' % (i, j, test_cost, err*100))
					costs.append(test_cost)
	print('Elapsed time:', (datetime.now() - t0))
	plt.plot(costs, label='test set')
	plt.title('Cross-Entropy Cost')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
