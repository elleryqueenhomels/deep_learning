# ANN in TensorFlow, with: batch SGD, RMSprop, Nesterov momentum, dropout regularization

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
	def __init__(self, hidden_layer_sizes, p_keep, activation_type=1):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.dropout_rates = np.array(p_keep, dtype=np.float32)
		self.activation_type = activation_type

	def fit(self, X, Y, epochs=10000, batch_sz=0, learning_rate=10e-6, decay=0, momentum=0, debug=False, cal_train=False, debug_points=100, valid_set=None):
		lr = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(momentum)

		# pre-process X, Y
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		if len(Y.shape) == 2:
			if Y.shape[1] == 1:
				Y = np.squeeze(Y)
			else:
				Y = np.argmax(Y, axis=1)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)

		N, D = X.shape
		K = len(set(Y))

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
						Xvalid, Yvalid = valid_set[0], np.squeeze(valid_set[1])
						if len(Yvalid.shape) == 2:
							Yvalid = np.argmax(Yvalid, axis=1)
						Xvalid = Xvalid.astype(np.float32)
						Yvalid = Yvalid.astype(np.int32)
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
		self.tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
		self.tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')
		pY = self.th_forward(self.tfX)
		self.forward_op = pY

		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pY, labels=self.tfY))
		prediction = self.th_predict(self.tfX)
		self.predict_op = prediction

		train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

		if debug:
			costs_train, costs_valid = [], []
			scores_train, scores_valid = [], []

		init = tf.global_variables_initializer()
		self.session = tf.Session()
		self.session.run(init)

		if batch_sz > 0 and batch_sz < N:
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

					self.session.run(train_op, feed_dict={self.tfX: Xbatch, self.tfY: Ybatch})

					# for debug:
					if debug:
						if i % print_epoch == 0 and j % print_batch == 0:
							if cal_train:
								ctrain = self.session.run(cost, feed_dict={self.tfX: X, self.tfY: Y})
								pYtrain = self.session.run(prediction, feed_dict={self.tfX: X})
								strain = classification_rate(Y, pYtrain)
								costs_train.append(ctrain)
								scores_train.append(strain)
								print('epoch=%d, batch=%d, n_batches=%d: cost_train=%s, score_train=%.6f%%' % (i, j, n_batches, ctrain, strain*100))
							if valid_set is not None:
								cvalid = self.session.run(cost, feed_dict={self.tfX: Xvalid, self.tfY: Yvalid})
								pYvalid = self.session.run(prediction, feed_dict={self.tfX: Xvalid})
								svalid = classification_rate(Yvalid, pYvalid)
								costs_valid.append(cvalid)
								scores_valid.append(svalid)
								print('epoch=%d, batch=%d, n_batches=%d: cost_valid=%s, score_valid=%.6f%%' % (i, j, n_batches, cvalid, svalid*100))
		else:
			# training: Backpropagation, using full gradient descent
			if debug:
				print_epoch = max(int(epochs / debug_points), 1)

			for i in range(epochs):
				self.session.run(train_op, feed_dict={self.tfX: X, self.tfY: Y})

				# for debug:
				if debug:
					if i % print_epoch == 0:
						if cal_train:
							ctrain = self.session.run(cost, feed_dict={self.tfX: X, self.tfY: Y})
							pYtrain = self.session.run(prediction, feed_dict={self.tfX: X})
							strain = classification_rate(Y, pYtrain)
							costs_train.append(ctrain)
							scores_train.append(strain)
							print('epoch=%d: cost_train=%s, score_train=%.6f%%' % (i, ctrain, strain*100))
						if valid_set is not None:
							cvalid = self.session.run(cost, feed_dict={self.tfX: Xvalid, self.tfY: Yvalid})
							pYvalid = self.session.run(prediction, feed_dict={self.tfX: Xvalid})
							svalid = classification_rate(Yvalid, pYvalid)
							costs_valid.append(cvalid)
							scores_valid.append(svalid)
							print('epoch=%d: cost_valid=%s, score_valid=%.6f%%' % (i, cvalid, svalid*100))

		if debug:
			if cal_train:
				ctrain = self.session.run(cost, feed_dict={self.tfX: X, self.tfY: Y})
				pYtrain = self.session.run(prediction, feed_dict={self.tfX: X})
				strain = classification_rate(Y, pYtrain)
				costs_train.append(ctrain)
				scores_train.append(strain)
				print('Final validation: cost_train=%s, score_train=%.6f%%, train_size=%d' % (ctrain, strain*100, len(Y)))
			if valid_set is not None:
				cvalid = self.session.run(cost, feed_dict={self.tfX: Xvalid, self.tfY: Yvalid})
				pYvalid = self.session.run(prediction, feed_dict={self.tfX: Xvalid})
				svalid = classification_rate(Yvalid, pYvalid)
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
		for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
			Z = tf.nn.dropout(Z, p)
			Z = h.forward(Z)
		Z = tf.nn.dropout(Z, self.dropout_rates[-1])
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

