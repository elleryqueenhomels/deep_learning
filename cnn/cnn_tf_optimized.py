# CNN in TensorFlow, with: mini-batch SGD, RMSprop, Nesterov Momentum, L2 Regularization

import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
# from util import shuffle  # If no sklearn installed, using this shuffle() instead.


class ConvPoolLayer(object):
	def __init__(self, mi, mo, fw, fh, poolsz=(2, 2), activation_type=1):
		# mi = number of input feature maps
		# mo = number of output feature maps
		# fw = filter width
		# fh = filter height
		shape = (fw, fh, mi, mo)
		W = init_filter(shape, poolsz)
		b = np.zeros(mo, dtype=np.float32)
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)
		self.poolsz = poolsz
		self.params = [self.W, self.b]
		self.activation = get_activation(activation_type)

	def forward(self, X):
		# X.shape = (N, xw, xh, c)
		# W.shape = (fw, fh, mi, mo) # W is filters, and mi == c
		# Y.shape = (N, yw, yh, mo) # Y is conv_out
		# By default in TensorFlow conv2d() operation:
		# yw = xw
		# yh = xh
		conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
		conv_out = tf.nn.bias_add(conv_out, self.b)
		pw, ph = self.poolsz
		pool_out = tf.nn.max_pool(conv_out, ksize=[1, pw, ph, 1], strides=[1, pw, ph, 1], padding='SAME')
		# pool_out.shape = (N, outw, outh, mo)
		# outw = int(yw / poolsz[0])
		# outh = int(yh / poolsz[1])
		# b.shape = (mo,)
		return self.activation(pool_out)


class HiddenLayer(object):
	def __init__(self, M1, M2, activation_type=1):
		W, b = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)
		self.params = [self.W, self.b]
		self.activation = get_activation(activation_type)

	def forward(self, X):
		return self.activation(tf.matmul(X, self.W) + self.b)


class CNN(object):
	def __init__(self, conv_layer_sizes, hidden_layer_sizes, pool_layer_sizes=None, convpool_activation=1, hidden_activation=1):
		self.conv_layer_sizes = conv_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes
		if pool_layer_sizes is None:
			pool_layer_sizes = [(2, 2) for i in range(len(conv_layer_sizes))]
		self.pool_layer_sizes = pool_layer_sizes
		assert(len(conv_layer_sizes) == len(pool_layer_sizes))
		self.convpool_activation = convpool_activation
		self.hidden_activation = hidden_activation

	def fit(self, X, Y, epochs=1000, batch_sz=100, learning_rate=10e-6, decay=0, momentum=0, reg_l2=0, debug=False, cal_train=False, debug_points=100, valid_set=None):
		# use float32 for GPU mode
		lr = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(momentum)
		reg = np.float32(reg_l2)

		# train set pre-processing
		assert(len(X) == len(Y))
		if len(Y.shape) == 1:
			Y = y2indicator(Y)
		elif Y.shape[1] == 1:
			Y = y2indicator(np.squeeze(Y))
		X = X.astype(np.float32)
		Y = Y.astype(np.float32)

		# get the number of labels
		K = Y.shape[1]

		# for debug: pre-process validation set
		if debug:
			if valid_set is not None:
				if len(valid_set) < 2 or len(valid_set[0]) != len(valid_set[1]) or len(valid_set[0].shape) != 4:
					valid_set = None
				else:
					Xvalid, Yvalid = valid_set[0], valid_set[1]
					if Xvalid.shape[1] != X.shape[1] or Xvalid.shape[2] != X.shape[2] or Xvalid.shape[3] != X.shape[3]:
						valid_set = None
					else:
						if len(Yvalid.shape) == 1:
							Yvalid = y2indicator(Yvalid, K)
						elif Yvalid.shape[1] == 1:
							Yvalid = y2indicator(np.squeeze(Yvalid), K)
						Xvalid = Xvalid.astype(np.float32)
						Yvalid = Yvalid.astype(np.float32)
						Yvalid_flat = np.argmax(Yvalid, axis=1)
			debug = cal_train or (valid_set is not None)

		# initialize convpool layers
		N, width, height, c = X.shape
		mi = c
		outw = width
		outh = height
		self.convpool_layers = []
		for convsz, poolsz in zip(self.conv_layer_sizes, self.pool_layer_sizes):
			mo, fw, fh = convsz
			cp = ConvPoolLayer(mi, mo, fw, fh, poolsz, activation_type=self.convpool_activation)
			self.convpool_layers.append(cp)
			outw = int(outw / poolsz[0])
			outh = int(outh / poolsz[1])
			mi = mo

		# initialize hidden layers
		self.hidden_layers = []
		M1 = mi * outw * outh # Here, mi == self.conv_layer_sizes[-1][0]
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, activation_type=self.hidden_activation)
			self.hidden_layers.append(h)
			M1 = M2

		# last output layer -- logistic regression layer
		W, b = init_weight_and_bias(M1, K)
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params
		for cp in reversed(self.convpool_layers):
			self.params += cp.params

		# set up tensorflow variables and functions
		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = int(N / 100)

		tfX = tf.placeholder(tf.float32, shape=(batch_sz, width, height, c), name='X') # Used for training, in order to avoid RAM swapping frequently
		tfY = tf.placeholder(tf.float32, shape=(batch_sz, K), name='Y') # Used for training, in order to avoid RAM swapping frequently
		self.tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X') # Used for 'forward' and 'predict', after trained the model

		pY = self.th_forward(tfX)
		self.forward_op = self.th_forward(self.tfX)

		reg_cost = reg * sum([tf.nn.l2_loss(p) for p in self.params])
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=tfY)) + reg_cost

		prediction = self.th_predict(tfX)
		self.predict_op = self.th_predict(self.tfX)

		train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

		if debug:
			costs_train, costs_valid = [], []
			scores_train, scores_valid = [], []

		init = tf.global_variables_initializer()
		self.session = tf.Session()
		self.session.run(init)

		# training: Backpropagation, using mini-batch stochastic gradient descent
		n_batches = int(N / batch_sz)

		if debug:
			debug_points = np.sqrt(debug_points)
			print_epoch, print_batch = max(int(epochs / debug_points), 1), max(int(n_batches / debug_points), 1)

		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
				Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

				self.session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

				# for debug:
				if debug:
					if i % print_epoch == 0 and j % print_batch == 0:
						if cal_train:
							Ytrain = np.argmax(Y, axis=1)
							ctrain = 0
							train_length = batch_sz * n_batches
							pYtrain = np.zeros(train_length)
							for k in range(n_batches):
								Xtrainbatch = X[k*batch_sz:(k*batch_sz + batch_sz)]
								Ytrainbatch = Y[k*batch_sz:(k*batch_sz + batch_sz)]
								ctrain += self.session.run(cost, feed_dict={tfX: Xtrainbatch, tfY: Ytrainbatch})
								pYtrain[k*batch_sz:(k*batch_sz + batch_sz)] = self.session.run(prediction, feed_dict={tfX: Xtrainbatch})
							strain = classification_rate(Ytrain[:train_length], pYtrain)
							costs_train.append(ctrain)
							scores_train.append(strain)
							print('epoch=%d, batch=%d, n_batches=%d: cost_train=%s, score_train=%.6f%%' % (i, j, n_batches, ctrain, strain*100))
						if valid_set is not None:
							cvalid = 0
							valid_length = batch_sz * int(len(Yvalid) / batch_sz)
							pYvalid = np.zeros(valid_length)
							for k in range(int(len(Yvalid) / batch_sz)):
								Xvalidbatch = Xvalid[k*batch_sz:(k*batch_sz + batch_sz)]
								Yvalidbatch = Yvalid[k*batch_sz:(k*batch_sz + batch_sz)]
								cvalid += self.session.run(cost, feed_dict={tfX: Xvalidbatch, tfY: Yvalidbatch})
								pYvalid[k*batch_sz:(k*batch_sz + batch_sz)] = self.session.run(prediction, feed_dict={tfX: Xvalidbatch})
							svalid = classification_rate(Yvalid_flat[:valid_length], pYvalid)
							costs_valid.append(cvalid)
							scores_valid.append(svalid)
							print('epoch=%d, batch=%d, n_batches=%d: cost_valid=%s, score_valid=%.6f%%' % (i, j, n_batches, cvalid, svalid*100))

		if debug:
			if cal_train:
				Ytrain = np.argmax(Y, axis=1)
				ctrain = 0
				train_length = batch_sz * n_batches
				pYtrain = np.zeros(train_length)
				for k in range(n_batches):
					Xtrainbatch = X[k*batch_sz:(k*batch_sz + batch_sz)]
					Ytrainbatch = Y[k*batch_sz:(k*batch_sz + batch_sz)]
					ctrain += self.session.run(cost, feed_dict={tfX: Xtrainbatch, tfY: Ytrainbatch})
					pYtrain[k*batch_sz:(k*batch_sz + batch_sz)] = self.session.run(prediction, feed_dict={tfX: Xtrainbatch})
				strain = classification_rate(Ytrain[:train_length], pYtrain)
				costs_train.append(ctrain)
				scores_train.append(strain)
				print('Final validation: cost_train=%s, score_train=%.6f%%, train_size=%d' % (ctrain, strain*100, len(Y)))
			if valid_set is not None:
				cvalid = 0
				valid_length = batch_sz * int(len(Yvalid) / batch_sz)
				pYvalid = np.zeros(valid_length)
				for k in range(int(len(Yvalid) / batch_sz)):
					Xvalidbatch = Xvalid[k*batch_sz:(k*batch_sz + batch_sz)]
					Yvalidbatch = Yvalid[k*batch_sz:(k*batch_sz + batch_sz)]
					cvalid += self.session.run(cost, feed_dict={tfX: Xvalidbatch, tfY: Yvalidbatch})
					pYvalid[k*batch_sz:(k*batch_sz + batch_sz)] = self.session.run(prediction, feed_dict={tfX: Xvalidbatch})
				svalid = classification_rate(Yvalid_flat[:valid_length], pYvalid)
				costs_valid.append(cvalid)
				scores_valid.append(svalid)
				print('Final validation: cost_valid=%s, score_valid=%.6f%%, valid_size=%d' % (cvalid, svalid*100, len(Yvalid)))

			import matplotlib.pyplot as plt
			if cal_train:
				plt.plot(costs_train, label='training set')
			if valid_set is not None:
				plt.plot(costs_valid, label='validation set')
			plt.title('Cross-Entropy Cost')
			plt.legend()
			plt.show()
			if cal_train:
				plt.plot(scores_train, label='training set')
			if valid_set is not None:
				plt.plot(scores_valid, label='validation set')
			plt.title('Classification Rate')
			plt.legend()
			plt.show()

	def th_forward(self, X):
		Z = X
		for cp in self.convpool_layers:
			Z = cp.forward(Z)
		Z_shape = Z.get_shape().as_list()
		Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def th_predict(self, X):
		pY = self.th_forward(X)
		return tf.argmax(pY, 1)

	def forward(self, X):
		X = X.astype(np.float32)
		return self.session.run(self.forward_op, feed_dict={self.tfX: X})

	def predict(self, X):
		X = X.astype(np.float32)
		return self.session.run(self.predict_op, feed_dict={self.tfX: X})

	def score(self, X, Y):
		P = self.predict(X)
		return classification_rate(Y, P)


def init_filter(shape, poolsz):
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return W.astype(np.float32)


def init_weight_and_bias(M1, M2):
	M1, M2 = int(M1), int(M2)
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def get_activation(activation_type):
	if activation_type == 1:
		return tf.nn.relu
	elif activation_type == 2:
		return tf.tanh
	elif activation_type == 3:
		return tf.nn.elu
	else:
		return tf.nn.sigmoid


def classification_rate(T, P):
	return np.mean(T == P)


def y2indicator(Y, K=None):
	N = len(Y)
	if K == None:
		K = len(set(Y))
	T = np.zeros((N, K))
	T[np.arange(N), Y.astype(np.int32)] = 1
	return T

