# Deep AutoEncoder Visualization --> Architecture shape: X-Wing

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weights, get_mnist

# new additions used to compare purity measure using GMM
import os
import sys
sys.path.append(os.path.abspath('..'))
from unsupervised_ml.k_means_evaluation import purity
# from unsupervised_ml.gmm import GMM
from sklearn.mixture import GaussianMixture


class Layer(object):
	def __init__(self, M1, M2):
		W = init_weights((M1, M2))
		bi = np.zeros(M2, dtype=np.float32)
		bo = np.zeros(M1, dtype=np.float32)
		self.W = theano.shared(W)
		self.bi = theano.shared(bi)
		self.bo = theano.shared(bo)
		self.params = [self.W, self.bi, self.bo]

	def forward(self, X):
		return T.nnet.sigmoid(X.dot(self.W) + self.bi)

	def forward_transpose(self, X):
		return T.nnet.sigmoid(X.dot(self.W.T) + self.bo)


class DeepAutoEncoder(object):
	def __init__(self, hidden_layer_sizes):
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, epochs=50, batch_sz=100, learning_rate=0.5, momentum=0.99, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		N, D = X.shape
		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N // 100
		n_batches = N // batch_sz

		# initialize hidden layers
		self.layers = []
		self.params = []
		mi = D
		for mo in self.hidden_layer_sizes:
			layer = Layer(mi, mo)
			self.layers.append(layer)
			self.params += layer.params
			mi = mo

		X_in = T.fmatrix('X')
		X_hat = self.forward(X_in)

		# cost = T.mean((X_in - X_hat) * (X_in - X_hat))
		cost = -T.mean(X_in * T.log(X_hat) + (one - X_in) * T.log(one - X_hat))
		cost_op = theano.function(inputs=[X_in], outputs=cost)

		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
		grads = T.grad(cost, self.params)

		updates = []
		for p, g, dp in zip(self.params, grads, dparams):
			updates += [
				(p, p + mu*dp - lr*g),
				(dp, mu*dp - lr*g)
			]

		train_op = theano.function(inputs=[X_in], outputs=cost, updates=updates)

		costs = []
		for i in range(epochs):
			print('epoch %d:' % i)
			X = shuffle(X)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
				c = train_op(Xbatch)
				costs.append(c)
				if j % 10 == 0:
					print('batch / n_batches: %d / %d, cost: %.6f' % (j, n_batches, c))

		if show_fig:
			plt.plot(costs)
			plt.title('Costs in Deep AutoEncoder')
			plt.show()

	def forward(self, X):
		Z = X
		for layer in self.layers:
			Z = layer.forward(Z)

		self.map2center = theano.function(inputs=[X], outputs=Z)

		for layer in reversed(self.layers):
			Z = layer.forward_transpose(Z)

		return Z


def main():
	Xtrain, Ytrain, _, _ = get_mnist(42000)
	dae = DeepAutoEncoder([500, 300, 2])
	dae.fit(Xtrain, show_fig=True)
	mapping = dae.map2center(Xtrain)
	plt.scatter(mapping[:,0], mapping[:,1], c=Ytrain, s=50, alpha=0.5)
	plt.show()

	# purity measure from unsupervised machine learning
	print('\nPurity measure via GMM:')
	gmm = GaussianMixture(n_components=10)
	gmm.fit(Xtrain) # this may be very slow.
	responsibilities_full = gmm.predict_proba(Xtrain)
	print('full purity: %s' % purity(Ytrain, responsibilities_full))

	gmm.fit(mapping)
	responsibilities_reduced = gmm.predict_proba(mapping)
	print('reduced purity: %s' % purity(Ytrain, responsibilities_reduced))


if __name__ == '__main__':
	main()

