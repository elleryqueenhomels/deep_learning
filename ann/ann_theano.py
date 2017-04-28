# ANN in Theano, with: batch SGD, RMSprop, Nesterov momentum, L2 regularization (No dropout regularization)
# A GPU accelerated Theano code version, using float32
# THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python3 ann_theano.py
# Or just simply use this in the code file:
# import os
# os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,floatX=float32,device=gpu'
# import theano

import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import shuffle # If no sklean installed, just use: _shuffle


class HiddenLayer(object):
	def __init__(self, M1, M2, activation_type=1):
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.params = [self.W, self.b]
		self.activation = get_activation(activation_type)

	def forward(self, X):
		return self.activation(X.dot(self.W) + self.b)


class ANN(object):
	def __init__(self, hidden_layer_sizes, activation_type=1):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation_type = activation_type

	def fit(self, X, Y, epochs=10000, batch_sz=0, learning_rate=10e-6, decay=0, momentum=0, reg_l2=0, eps=10e-10, debug=False, cal_train=False, debug_points=100, valid_set=None):
		# for GPU accelerated, using float32
		lr = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(momentum)
		reg = np.float32(reg_l2)
		eps = np.float32(eps)
		one = np.float32(1)

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
		N, D = X.shape
		K = len(set(Y))

		self.hidden_layers = []
		M1 = D
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, self.activation_type)
			self.hidden_layers.append(h)
			M1 = M2
		W, b = init_weight_and_bias(M1, K)
		self.W = theano.shared(W)
		self.b = theano.shared(b)

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params

		# set up theano variables and functions
		thX = T.fmatrix('X')
		thY = T.ivector('Y')
		pY = self.th_forward(thX)
		self.forward_op = theano.function(inputs=[thX], outputs=pY)

		reg_cost = reg * T.sum([(p * p).sum() for p in self.params])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + reg_cost
		prediction = self.th_predict(thX)
		self.predict_op = theano.function(inputs=[thX], outputs=prediction)

		# cost and prediction operation
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		updates = []
		if decay > 0 and decay < 1:
			# for RMSprop
			cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
			if mu > 0:
				# for momentum
				dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
				for c, dp, p in zip(cache, dparams, self.params):
					updates += [
						(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)),
						(p, p + mu*mu*dp - (one + mu)*lr*T.grad(cost, p)/T.sqrt(c + eps)),
						(dp, mu*mu*dp - (one + mu)*lr*T.grad(cost, p)/T.sqrt(c + eps))
					]
			else:
				for c, p in zip(cache, self.params):
					updates += [
						(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)),
						(p, p - lr*T.grad(cost, p)/T.sqrt(c + eps))
					]
		else:
			if mu > 0:
				# for momentum
				dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
				for dp, p in zip(dparams, self.params):
					updates += [
						(p, p + mu*mu*dp - (one + mu)*lr*T.grad(cost, p)),
						(dp, mu*mu*dp - (one + mu)*lr*T.grad(cost, p))
					]
			else:
				updates = [(p, p - lr*T.grad(cost, p)) for p in self.params]

		train_op = theano.function(inputs=[thX, thY], updates=updates)

		if debug:
			costs_train, costs_valid = [], []
			scores_train, scores_valid = [], []

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

					train_op(Xbatch, Ybatch)

					# for debug:
					if debug:
						if i % print_epoch == 0 and j % print_batch == 0:
							if cal_train:
								ctrain, pYtrain = cost_predict_op(X, Y)
								strain = classification_rate(Y, pYtrain)
								costs_train.append(ctrain)
								scores_train.append(strain)
								print('epoch=%d, batch=%d, n_batches=%d: cost_train=%s, score_train=%.6f%%' % (i, j, n_batches, ctrain, strain*100))
							if valid_set is not None:
								cvalid, pYvalid = cost_predict_op(Xvalid, Yvalid)
								svalid = classification_rate(Yvalid, pYvalid)
								costs_valid.append(cvalid)
								scores_valid.append(svalid)
								print('epoch=%d, batch=%d, n_batches=%d: cost_valid=%s, score_valid=%.6f%%' % (i, j, n_batches, cvalid, svalid*100))
		else:
			# training: Backpropagation, using full gradient descent
			if debug:
				print_epoch = max(int(epochs / debug_points), 1)

			for i in range(epochs):
				train_op(X, Y)

				# for debug:
				if debug:
					if i % print_epoch == 0:
						if cal_train:
							ctrain, pYtrain = cost_predict_op(X, Y)
							strain = classification_rate(Y, pYtrain)
							costs_train.append(ctrain)
							scores_train.append(strain)
							print('epoch=%d: cost_train=%s, score_train=%.6f%%' % (i, ctrain, strain*100))
						if valid_set is not None:
							cvalid, pYvalid = cost_predict_op(Xvalid, Yvalid)
							svalid = classification_rate(Yvalid, pYvalid)
							costs_valid.append(cvalid)
							scores_valid.append(svalid)
							print('epoch=%d: cost_valid=%s, score_valid=%.6f%%' % (i, cvalid, svalid*100))

		if debug:
			if cal_train:
				ctrain, pYtrain = cost_predict_op(X, Y)
				strain = classification_rate(Y, pYtrain)
				costs_train.append(ctrain)
				scores_train.append(strain)
				print('Final validation: cost_train=%s, score_train=%.6f%%, train_size=%d' % (ctrain, strain*100, len(Y)))
			if valid_set is not None:
				cvalid, pYvalid = cost_predict_op(Xvalid, Yvalid)
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
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def th_predict(self, X):
		pY = self.th_forward(X)
		return T.argmax(pY, axis=1)

	def forward(self, X):
		X = X.astype(np.float32)
		return self.forward_op(X)

	def predict(self, X):
		X = X.astype(np.float32)
		return self.predict_op(X)

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
		return T.nnet.relu
	elif activation_type == 2:
		return T.tanh
	elif activation_type == 3:
		return elu
	else:
		return T.nnet.sigmoid

def elu(X):
	return T.switch(X >= np.float32(0), X, (T.exp(X) - np.float32(1)))

def classification_rate(targets, predictions):
	return np.mean(targets == predictions)

# just in case you have not installed sklearn
def _shuffle(X, Y):
	assert(len(X) == len(Y))
	idx = np.arange(len(Y))
	np.random.shuffle(idx)
	return X[idx], Y[idx]

