# ANN in TensorFlow, with: batch SGD, RMSprop, Nesterov momentum, L2 regularization (No dropout regularization)

import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle # If no sklean installed, just use: _shuffle


class HiddenLayer(object):
	def __init__(self, M1, M2, activation_type=1):
		W, b = init_weight_and_bias(M1, M2)
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)
		self.params = [self.W, self.b]
		self.activation = get_activation(activation_type)

	def forward(self, X):
		return self.activation(tf.matmul(X, self.W) + self.b)


class ANN(object):
	def __init__(self, hidden_layer_sizes, activation_type=1):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation_type = activation_type

	def fit(self, X, Y, epochs=10000, batch_sz=0, learning_rate=10e-6, decay=0, momentum=0, reg_l2=0, debug=False, cal_train=False, debug_points=100, valid_set=None):
		lr = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(momentum)
		reg = np.float32(reg_l2)

		# pre-process X, Y
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		if len(Y.shape) == 1:
			Y = y2indicator(Y)
		elif Y.shape[1] == 1:
			Y = y2indicator(np.squeeze(Y))
		X = X.astype(np.float32)
		Y = Y.astype(np.float32)

		N, D = X.shape
		K = Y.shape[1]

		# for debug: pre-process validation set
		if debug:
			if valid_set is not None:
				if len(valid_set) < 2 or len(valid_set[0]) != len(valid_set[1]):
					valid_set = None
				else:
					if len(valid_set[0].shape) == 1:
						valid_set[0] = valid_set[0].reshape(-1, 1)
					if valid_set[0].shape[1] != X.shape[1]:
						valid_set = None
					else:
						Xvalid, Yvalid = valid_set[0], valid_set[1]
						if len(Yvalid.shape) == 1:
							Yvalid = y2indicator(Yvalid, K)
						elif Yvalid.shape[1] == 1:
							Yvalid = y2indicator(np.squeeze(Yvalid), K)
						Xvalid = Xvalid.astype(np.float32)
						Yvalid = Yvalid.astype(np.float32)
						Yvalid_flat = np.argmax(Yvalid, axis=1)
			debug = cal_train or (valid_set is not None)

		# initialize hidden layers
		self.hidden_layers = []
		M1 = D
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, self.activation_type)
			self.hidden_layers.append(h)
			M1 = M2
		W, b = init_weight_and_bias(M1, K)
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params

		# set up theano variables and functions
		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = int(N / 100)

		tfX = tf.placeholder(tf.float32, shape=(batch_sz, D), name='X') # Used for training, in order to avoid RAM swapping frequently
		tfY = tf.placeholder(tf.float32, shape=(batch_sz, K), name='Y') # Used for training, in order to avoid RAM swapping frequently
		self.tfX = tf.placeholder(tf.float32, shape=(None, D), name='X') # Used for 'forward' and 'predict', after trained the model

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

		# training: Backpropagation, using batch gradient descent
		n_batches = int(N / batch_sz)
		if debug:
			debug_points = np.sqrt(debug_points)
			print_epoch, print_batch = max(int(epochs / debug_points), 1), max(int(n_batches / debug_points), 1)

		for i in range(epochs):
			X, Y = shuffle(X, Y) # if no sklearn, just use: _shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
				Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

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
		pY = self.predict(X)
		return np.mean(Y == pY)


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

def classification_rate(targets, predictions):
	return np.mean(targets == predictions)

def y2indicator(Y, K=None):
	N = len(Y)
	if K == None:
		K = len(set(Y))
	T = np.zeros((N, K))
	T[np.arange(N), Y.astype(np.int32)] = 1
	return T

# just in case you have not installed sklearn
def _shuffle(X, Y):
	assert(len(X) == len(Y))
	idx = np.arange(len(Y))
	np.random.shuffle(idx)
	return X[idx], Y[idx]

