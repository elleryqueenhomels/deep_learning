# Deep Neural Network with AutoEncoder Pretraining

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle


class AutoEncoder(object):
	def __init__(self, M, an_id=None, activation_type=1, cost_type=1):
		self.M = M
		self.id = an_id
		self.cost_type = cost_type
		if activation_type == 1:
			self.activation = T.nnet.sigmoid
		elif activation_type == 2:
			self.activation = T.tanh
		else:
			self.activation = T.nnet.relu

	def fit(self, X, epochs=1, batch_sz=100, learning_rate=0.5, momentum=0.99, debug=False, print_period=20, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		X = X.astype(np.float32)
		N, D = X.shape

		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N
		n_batches = N // batch_sz

		W = init_weights((D, self.M))
		bh = np.zeros(self.M, dtype=np.float32)
		bo = np.zeros(D, dtype=np.float32)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.bh = theano.shared(bh, 'bh_%s' % self.id)
		self.bo = theano.shared(bo, 'bo_%s' % self.id)
		self.params = [self.W, self.bh, self.bo]
		self.forward_params = [self.W, self.bh]

		# for Momentum
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		X_in = T.fmatrix('X_%s' % self.id)
		X_hat = self.forward_output(X_in)

		H = self.forward_hidden(X_in)
		self.forward_hidden_op = theano.function(inputs=[X_in], outputs=H)

		if self.cost_type == 1:
			cost = -T.mean(X_in * T.log(X_hat) + (one - X_in) * T.log(one - X_hat))
		else:
			cost = T.mean((X_in - X_hat) * (X_in - X_hat))
		cost_op = theano.function(inputs=[X_in], outputs=cost)

		updates = []
		for p, dp in zip(self.params, dparams):
			updates += [
				(p, p + mu*dp - lr*T.grad(cost, p)),
				(dp, mu*dp - lr*T.grad(cost, p))
			]

		train_op = theano.function(inputs=[X_in], updates=updates)

		if debug:
			print('\nTraining AutoEncoder %s:' % self.id)
			costs = []
		for i in range(epochs):
			X = shuffle(X)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]

				train_op(Xbatch)

				if debug and j % print_period == 0:
					the_cost = cost_op(X)
					costs.append(the_cost)
					print('epoch=%d, batch=%d, n_batches=%d: cost=%.6f' % (i, j, n_batches, the_cost))

		if debug:
			the_cost = cost_op(X)
			costs.append(the_cost)
			print('Finally: cost=%.6f' % (the_cost))

			if show_fig:
				plt.plot(costs)
				plt.title('Costs in AutoEncoder %s' % self.id)
				plt.show()

	def forward_hidden(self, X):
		return self.activation(X.dot(self.W) + self.bh)

	def forward_output(self, X):
		Z = self.forward_hidden(X)
		return self.activation(Z.dot(self.W.T) + self.bo)

	@staticmethod
	def createFromArray(W, bh, bo, an_id=None, activation_type=1, cost_type=1):
		ae = AutoEncoder(W.shape[1], an_id, activation_type, cost_type)
		ae.W = theano.shared(W.astype(np.float32), 'W_%s' % ae.id)
		ae.bh = theano.shared(bh.astype(np.float32), 'bh_%s' % ae.id)
		ae.bo = theano.shared(bo.astype(np.float32), 'bo_%s' % ae.id)
		ae.params = [ae.W, ae.bh, ae.bo]
		ae.forward_params = [ae.W, ae.bh]
		return ae


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

		# greedy layer-wise training of autoencoders
		if not pretrain:
			pretrain_epochs = 0

		current_input = X
		for ae in self.hidden_layers:
			ae.fit(current_input, epochs=pretrain_epochs, batch_sz=pretrain_batch_sz, debug=debug, print_period=print_period, show_fig=show_fig)
			current_input = ae.forward_hidden_op(current_input) # create current_input for next layer

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

		updates = []
		for p, dp in zip(self.params, dparams):
			updates += [
				(p, p + mu*dp - lr*T.grad(cost, p)),
				(dp, mu*dp - lr*T.grad(cost, p))
			]

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
			Z = h.forward_hidden(Z)
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


def init_weights(shape):
	W = np.random.randn(*shape) / np.sqrt(sum(shape))
	return W.astype(np.float32)


def classification_rate(T, P):
	return np.mean(T == P)


def preprocess(X, Y, debug):
	if debug:
		if X is None or Y is None or len(X) != len(Y):
			return None, None, False
	if len(X.shape) == 1:
		X = X.reshape(-1, 1)
	if len(Y.shape) == 2:
		if Y.shape[1] == 1:
			Y = np.squeeze(Y)
		else:
			Y = np.argmax(Y, axis=1)
	X = X.astype(np.float32)
	Y = Y.astype(np.int32)
	return X, Y, debug

