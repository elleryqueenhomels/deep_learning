# ANN in Theano, with: batch SGD, RMSprop, Nesterov Momentum, L2 regularization (No dropout regularization)
# A GPU accelerated Theano code version
# use GPU to run, command:
# sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 time python3 ann_theano.py
# Here, since there are some problems im my mac, I should use device=cpu instead of device=gpu:
# sudo THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 time python3 ann_theano.py
# or just simply use this in the *.py file:
# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano

import numpy as np
import theano
import theano.tensor as T

from util import getData, getBinaryData
from sklearn.utils import shuffle
from datetime import datetime


class HiddenLayer(object):
	def __init__(self, M1, M2, an_id=''):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(M1, M2)
		self.W = theano.shared(W, name='W_%s' % self.id)
		self.b = theano.shared(b, name='b_%s' % self.id)
		self.params = [self.W, self.b]

	def forward(self, X):
		return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, epochs=300, batch_sz=500, learning_rate=10e-6, decay=0.99, mu=0.9, reg=10e-6, eps=1e-10, show_fig=False):
		# for GPU accelerated
		learning_rate = np.float32(learning_rate)
		decay = np.float32(decay)
		mu = np.float32(mu)
		reg = np.float32(reg)
		eps = np.float32(eps)
		one = np.float32(1)

		# make a validation set
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)
		Ntrain = int(len(Y) / 4)
		Xvalid, Yvalid = X[Ntrain:], Y[Ntrain:]
		X, Y = X[:Ntrain], Y[:Ntrain]

		# initialize hidden layers
		N, D = X.shape
		K = len(set(Y))
		self.hidden_layers = []
		M1 = D
		count = 0
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2
			count += 1
		W, b = init_weight_and_bias(M1, K)
		self.W = theano.shared(W, name='W_logreg')
		self.b = theano.shared(b, name='b_logreg')

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params

		# for RMSprop
		cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		# for momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		# set up theano functions and variables
		thX = T.fmatrix('X')
		thY = T.ivector('Y')
		pY = self.th_forward(thX)

		reg_cost = reg * T.sum([(p*p).sum() for p in self.params])
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + reg_cost
		prediction = self.th_predict(thX)

		# actual prediction function
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		updates = []
		for c, dp, p in zip(cache, dparams, self.params):
			updates += [
				(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)),
				(dp, mu*mu*dp - (one + mu)*learning_rate*T.grad(cost, p)/T.sqrt(c + eps)),
				(p, p + dp)
			]

		# another version of updates
		# updates = [
		# 	(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)) for p, c in zip(self.params, cache)
		# ] + [
		# 	(p, p + mu*mu*dp - (one + mu)*learning_rate*T.grad(cost, p)/T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
		# ] + [
		# 	(dp, mu*mu*dp - (one + mu)*learning_rate*T.grad(cost, p)/T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
		# ]

		# RMSprop only
		# updates = [
		# 	(c, decay*c + (one - decay)*T.grad(cost, p)*T.grad(cost, p)) for p, c in zip(self.params, cache)
		# ] + [
		# 	(p, p - learning_rate*T.grad(cost, p)/T.sqrt(c + eps)) for p, c in zip(self.params, cache)
		# ]

		# momentum only
		# updates = [
		# 	(p, p + mu*mu*dp - (one + mu)*learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
		# ] + [
		# 	(dp, mu*mu*dp - (one + mu)*learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
		# ]

		train_op = theano.function(inputs=[thX, thY], updates=updates)

		n_batches = int(N / batch_sz)
		costs = []
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
				Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

				train_op(Xbatch, Ybatch)

				if i % 20 == 0 and j % 10 == 0:
					c, p = cost_predict_op(Xvalid, Yvalid)
					costs.append(c)
					e = error_rate(Yvalid, p)
					print('epoch=%d, batch=%d, n_batches=%d: cost=%s, error rate=%.6f%%' % (i, j, n_batches, c, e*100))
		c, p = cost_predict_op(Xvalid, Yvalid)
		costs.append(c)
		e = error_rate(Yvalid, p)
		print('Final: cost=%s, error rate=%.6f%%, train size=%d' % (c, e*100, N))

		if show_fig:
			import matplotlib.pyplot as plt
			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()

	def th_forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return T.nnet.softmax(Z.dot(self.W) + self.b)

	def th_predict(self, X):
		pY = self.th_forward(X)
		return T.argmax(pY, axis=1)

	def predict(self, X):
		X = X.astype(np.float32)
		thX = T.fmatrix('X')
		prediction = self.th_predict(thX)
		predict_op = theano.function(inputs=[thX], outputs=[prediction])
		return predict_op(X)


def init_weight_and_bias(M1, M2):
	M1, M2 = int(M1), int(M2)
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)

def error_rate(targets, predictions):
	return np.mean(targets != predictions)


def main():
	print('\nBegin to extract data...')
	t0 = datetime.now()
	X, Y = getData()
	print('\nRead data time:', (datetime.now() - t0))

	model = ANN([2000, 1000])
	t0 = datetime.now()
	model.fit(X, Y, epochs=100, batch_sz=500, learning_rate=10e-7, decay=0.99, mu=0.9, reg=0.01, show_fig=False)
	print('Train time:', (datetime.now() - t0))

	P = model.predict(X)
	print('\nPredict attempt: score=%.6f%%\n' % ((1 - error_rate(Y, P)) * 100))


if __name__ == '__main__':
	main()
