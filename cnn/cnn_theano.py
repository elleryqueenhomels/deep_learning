# CNN in Theano, with: RMSprop, Nesterov Momentum, L2 Regularization

import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from sklearn.utils import shuffle


class ConvPoolLayer(object):
	def __init__(self, mi, mo, fw, fh, poolsz=(2, 2)):
		# mi = number of input feature maps
		# mo = number of output feature maps
		# fw = filter width
		# fh = filter height
		shape = (mo, mi, fw, fh)
		W = init_filter(shape, poolsz)
		b = np.zeros(mo, dtype=np.float32)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		# X.shape = (N, c, xw, xh)
		# W.shape = (mo, mi, fw, fh) # W is filters, and mi == c
		# Y.shape = (N, mo, yw, yh) # Y is conv_out
		# By default in Theano conv2d() operation:
		# yw = xw - fw + 1
		# yh = xh - fh + 1
		conv_out = conv2d(input=X, filters=self.W)
		pool_out = pool_2d(
			input=conv_out,
			ws=self.poolsz,
			ignore_border=True
		)
		# pool_out.shape = (N, mo, outw, outh)
		# outw = int(yw / poolsz[0])
		# outh = int(yh / poolsz[1])
		# b.shape = (mo,)
		# after dimshuffle(), new_b.shape = (1, mo, 1, 1)
		return T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class HiddenLayer(object):
	def __init__(self, M1, M2):
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.params = [self.W, self.b]

	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)


class CNN(object):
	def __init__(self, conv_layer_sizes, hidden_layer_sizes, pool_layer_sizes=None):
		self.conv_layer_sizes = conv_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes
		if pool_layer_sizes == None:
			pool_layer_sizes = [(2, 2) for i in range(len(conv_layer_sizes))]
		self.pool_layer_sizes = pool_layer_sizes
		assert(len(conv_layer_sizes) == len(pool_layer_sizes))

	def fit(self, X, Y, epochs=1000, batch_sz=0, learning_rate=10e-6, decay=0, momentum=0, reg_l2=0, eps=10e-10, debug=False, cal_train=False, debug_points=100, valid_set=None):
		# use float32 for Theano GPU mode
		lr = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(momentum)
		reg = np.float32(reg_l2)
		eps = np.float32(eps)
		one = np.float32(1)

		# train set pre-processing
		assert(len(X) == len(Y))
		if len(Y.shape) == 2:
			if Y.shape[1] == 1:
				Y = np.squeeze(Y)
			else:
				Y = np.argmax(Y, axis=1)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)

		# for debug: pre-process validation set
		if debug:
			if valid_set != None:
				if len(valid_set) < 2 or len(valid_set[0]) != len(valid_set[1]) or len(valid_set[0].shape) != 4:
					valid_set = None
				else:
					Xvalid = valid_set[0]
					if Xvalid.shape[1] != X.shape[1] or Xvalid.shape[2] != X.shape[2] or Xvalid.shape[3] != X.shape[3]:
						valid_set = None
					else:
						Yvalid = np.squeeze(valid_set[1])
						if len(Yvalid.shape) == 2:
							Yvalid = np.argmax(Yvalid, axis=1)
						Xvalid = Xvalid.astype(np.float32)
						Yvalid = Yvalid.astype(np.int32)
			debug = cal_train or (valid_set != None)

		# initialize convpool layers
		N, c, width, height = X.shape
		mi = c
		outw = width
		outh = height
		self.convpool_layers = []
		for convsz, poolsz in zip(self.conv_layer_sizes, self.pool_layer_sizes):
			mo, fw, fh = convsz
			cp = ConvPoolLayer(mi, mo, fw, fh, poolsz)
			self.convpool_layers.append(cp)
			outw = int((outw - fw + 1) / poolsz[0])
			outh = int((outh - fh + 1) / poolsz[1])
			mi = mo

		# initialize hidden layers
		K = len(set(Y))
		self.hidden_layers = []
		M1 = mi * outw * outh # Here, mi == self.conv_layer_sizes[-1][0]
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2)
			self.hidden_layers.append(h)
			M1 = M2

		# last output layer -- logistic regression layer
		W, b = init_weight_and_bias(M1, K)
		self.W = theano.shared(W)
		self.b = theano.shared(b)

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params
		for cp in reversed(self.convpool_layers):
			self.params += cp.params

		# for RMSprop
		cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		# for momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		# set up theano variables and functions
		thX = T.tensor4('X', dtype='float32')
		thY = T.ivector('Y') # lower case i means int32. By the way, upper case I means int64

		pY = self.th_forward(thX)
		self.forward_op = theano.function(inputs=[thX], outputs=pY)

		reg_cost = reg * T.sum([(p*p).sum() for p in self.params])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + reg_cost

		prediction = self.th_predict(thX)
		self.predict_op = theano.function(inputs=[thX], outputs=prediction)

		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		# updates with RMSprop and Momentum, optional
		updates = []
		if decay > 0 and decay < 1:
			if mu > 0:
				for c, dp, p in zip(cache, dparams, self.params):
					updates += [
						(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)),
						(dp, mu*mu*dp - (one + mu)*lr*T.grad(cost, p)/T.sqrt(c + eps)),
						(p, p + dp)
					]
			else:
				for c, p in zip(cache, self.params):
					updates += [
						(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)),
						(p, p - lr*T.grad(cost, p)/T.sqrt(c + eps))
					]
		else:
			if mu > 0:
				for dp, p in zip(dparams, self.params):
					updates += [
						(dp, mu*mu*dp - (one + mu)*lr*T.grad(cost, p)),
						(p, p + dp)
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
				X, Y = shuffle(X, Y)
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
					Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

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
							if valid_set != None:
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
						if valid_set != None:
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
			if valid_set != None:
				cvalid, pYvalid = cost_predict_op(Xvalid, Yvalid)
				svalid = classification_rate(Yvalid, pYvalid)
				costs_valid.append(cvalid)
				scores_valid.append(svalid)
				print('Final validation: cost_valid=%s, score_valid=%.6f%%, valid_size=%d' % (cvalid, svalid*100, len(Yvalid)))

			import matplotlib.pyplot as plt
			if cal_train:
				plt.plot(costs_train, label='training set')
			if valid_set != None:
				plt.plot(costs_valid, label='validation set')
			plt.title('Cross-Entropy Cost')
			plt.legend()
			plt.show()
			if cal_train:
				plt.plot(scores_train, label='training set')
			if valid_set != None:
				plt.plot(scores_valid, label='validation set')
			plt.title('Classification Rate')
			plt.legend()
			plt.show()

	def th_forward(self, X):
		Z = X
		for cp in self.convpool_layers:
			Z = cp.forward(Z)
		Z = Z.flatten(ndim=2)
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
		P = self.predict(X)
		return classification_rate(Y, P)


def init_filter(shape, poolsz):
	W = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return W.astype(np.float32)


def init_weight_and_bias(M1, M2):
	M1, M2 = int(M1), int(M2)
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def classification_rate(T, P):
	return np.mean(T == P)
