# Deep Neural Network with AutoEncoder/RBM Pretraining
# In general:
# Greedy Layer-wise AutoEncoder/RBM Pretraining is better than Pure Backpropagation
# Deep Stacked RBMs have a special name: Deep Belief Network (DBN)

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weights, preprocess, classification_rate
from autoencoder import AutoEncoder
from rbm import RBM


class DNN(object):
	def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder, activation_type=1, cost_type=1):
		self.hidden_layers = []
		count = 0
		for M in hidden_layer_sizes:
			h = UnsupervisedModel(M, count, activation_type, cost_type)
			self.hidden_layers.append(h)
			count += 1

	def fit(self, X, Y, Xtest=None, Ytest=None, epochs=1, batch_sz=100, pretrain=True, pretrain_epochs=1, pretrain_batch_sz=100, learning_rate=0.01, momentum=0.99, debug=False, print_period=20, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		# pre-processing
		X, Y, _ = preprocess(X, Y, False)
		Xtest, Ytest, debug = preprocess(Xtest, Ytest, debug)

		N, K = len(Y), len(set(Y))
		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N
		n_batches = N // batch_sz

		# greedy layer-wise pretraining of Unsupervised Model, e.g. AutoEncoder, RBM, etc.
		if not pretrain:
			pretrain_epochs = 0

		current_input = X
		for h in self.hidden_layers:
			h.fit(current_input, epochs=pretrain_epochs, batch_sz=pretrain_batch_sz, debug=debug, print_period=print_period, show_fig=show_fig)
			current_input = h.forward_hidden_op(current_input) # create current_input for next layer

		# initialize logistic regression layer
		W = init_weights((self.hidden_layers[-1].M, K))
		b = np.zeros(K, dtype=np.float32)
		self.W = theano.shared(W, 'W_logreg')
		self.b = theano.shared(b, 'b_logreg')

		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.forward_params

		# for momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		thX = T.fmatrix('X')
		thY = T.ivector('Y')

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

				if debug and j % print_period == 0:
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
				plt.title('Costs in DNN')
				plt.show()

	def th_forward(self, X):
		Z = X
		for h in self.hidden_layers:
			# Z = h.forward_hidden(Z)
			Z = h.forward(Z) # this 'forward' is the same as 'forward_hidden', just for compatibility and consistency.
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

