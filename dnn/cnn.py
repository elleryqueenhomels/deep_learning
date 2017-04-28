# Convolutional Neural Network with AutoEncoder/RBM Pretraining

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from sklearn.utils import shuffle
from util import init_filter, init_weight_and_bias, get_activation, preprocess_cnn, classification_rate
from autoencoder import AutoEncoder
from rbm import RBM


class ConvPoolLayer(object):
	def __init__(self, mi, mo, fw, fh, poolsz=(2, 2), activation_type=1):
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
		self.activation = get_activation(activation_type)

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
		return self.activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class CNN(object):
	def __init__(self, conv_layer_sizes, hidden_layer_sizes, pool_layer_sizes=None, UnsupervisedModel=AutoEncoder, convpool_activation=1, hidden_activation=1, hidden_cost=1):
		self.conv_layer_sizes = conv_layer_sizes
		if pool_layer_sizes is None:
			pool_layer_sizes = [(2, 2) for i in range(len(conv_layer_sizes))]
		self.pool_layer_sizes = pool_layer_sizes
		assert(len(conv_layer_sizes) == len(pool_layer_sizes))
		self.hidden_layers = []
		for i, M in enumerate(hidden_layer_sizes):
			h = UnsupervisedModel(M, i, activation_type=hidden_activation, cost_type=hidden_cost)
			self.hidden_layers.append(h)
		self.activation_type = convpool_activation

	def fit(self, X, Y, Xtest=None, Ytest=None, epochs=1, batch_sz=100, pretrain=True, pretrain_epochs=1, pretrain_batch_sz=100, pretrain_lr=0.5, pretrain_mu=0, learning_rate=0.01, momentum=0.99, debug=False, print_period=20, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		# pre-processing
		X, Y, _ = preprocess_cnn(X, Y, False)
		Xtest, Ytest, debug = preprocess_cnn(Xtest, Ytest, debug)

		N, K = len(Y), len(set(Y))
		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N
		n_batches = N // batch_sz

		# initialize convpool layers
		N, c, width, height = X.shape
		mi = c
		outw = width
		outh = height
		self.convpool_layers = []
		for convsz, poolsz in zip(self.conv_layer_sizes, self.pool_layer_sizes):
			mo, fw, fh = convsz
			cp = ConvPoolLayer(mi, mo, fw, fh, poolsz, activation_type=self.activation_type)
			self.convpool_layers.append(cp)
			outw = int((outw - fw + 1) / poolsz[0])
			outh = int((outh - fh + 1) / poolsz[1])
			mi = mo

		# initialize Theano variable/placeholder
		thX = T.tensor4('X', dtype='float32')
		thY = T.ivector('Y')

		# for UnsupervisedModel pretrain
		convpool_out = self.forward_convpool(thX)
		convpool_op = theano.function(inputs=[thX], outputs=convpool_out)

		# greedy layer-wise pretraining of Unsupervised Model, e.g. AutoEncoder, RBM, etc.
		if not pretrain:
			pretrain_epochs = 0

		current_input = convpool_op(X)
		for h in self.hidden_layers:
			h.fit(current_input, epochs=pretrain_epochs, batch_sz=pretrain_batch_sz, learning_rate=pretrain_lr, momentum=pretrain_mu, debug=debug, print_period=print_period, show_fig=show_fig)
			current_input = h.forward_hidden_op(current_input) # create current_input for next layer

		# initialize logistic regression layer
		W, b = init_weight_and_bias(self.hidden_layers[-1].M, K)
		self.W = theano.shared(W, 'W_logreg')
		self.b = theano.shared(b, 'b_logreg')

		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.forward_params
		for cp in reversed(self.convpool_layers):
			self.params += cp.params

		# for momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		pY = self.th_forward(thX)
		self.forward_op = theano.function(inputs=[thX], outputs=pY)

		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))

		prediction = self.th_predict(thX)
		self.predict_op = theano.function(inputs=[thX], outputs=prediction)

		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		# updates with momentum optional
		if mu > 0:
			updates = []
			for p, dp in zip(self.params, dparams):
				updates += [
					(p, p + mu*dp - lr*T.grad(cost, p)),
					(dp, mu*dp - lr*T.grad(cost, p))
				]
		else:
			updates = [(p, p - lr*T.grad(cost, p)) for p in self.params]

		train_op = theano.function(inputs=[thX, thY], updates=updates)

		if debug:
			print('\nSupervised training:')
			costs = []

		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
				Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

				train_op(Xbatch, Ybatch)

				if debug:
					if j % print_period == 0:
						the_cost, the_prediction = cost_predict_op(Xtest, Ytest)
						score = classification_rate(Ytest, the_prediction)
						costs.append(the_cost)
						print('epoch=%d, batch=%d, n_batches=%d: cost=%.6f, score=%.6f%%' % (i, j, n_batches, the_cost, score*100))

		if debug:
			the_cost, the_prediction = cost_predict_op(Xtest, Ytest)
			score = classification_rate(Ytest, the_prediction)
			costs.append(the_cost)
			print('Finally: cost=%.6f, score=%.6f%%' % (the_cost, score*100))

			if show_fig:
				plt.plot(costs)
				plt.title('Costs in CNN')
				plt.show()

	def forward_convpool(self, X):
		Z = X
		for cp in self.convpool_layers:
			Z = cp.forward(Z)
		Z = Z.flatten(ndim=2)
		return Z

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
		return np.mean(P == Y)

