# Deep Belief Network (DBN)
# This file is mainly used for visualizing what
# features a neuron in hidden layer has learned.

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import init_weights
from autoencoder import AutoEncoder
from rbm import RBM


class DBN(object):
	def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder, activation_type=1, cost_type=1):
		self.hidden_layers = []
		for i, M in enumerate(hidden_layer_sizes):
			h = UnsupervisedModel(M, i, activation_type, cost_type)
			self.hidden_layers.append(h)

	def fit(self, X, pretrain_epochs=1, pretrain_batch_sz=100, pretrain_lr=0.5, pretrain_mu=0, debug=False, print_period=20, show_fig=False):
		self.D = X.shape[1]

		current_input = X
		for h in self.hidden_layers:
			h.fit(current_input, epochs=pretrain_epochs, batch_sz=pretrain_batch_sz, learning_rate=pretrain_lr, momentum=pretrain_mu, debug=debug, print_period=print_period, show_fig=show_fig)
			current_input = h.forward_hidden_op(current_input)

		# return it here so we can use it directly after fitting without having to call forward()
		return current_input

	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			# Z = h.forward_hidden(Z)
			Z = h.forward(Z) # this 'forward' is the same as 'forward_hidden', just for compatibility and consistency.
		return Z

	# we assume hidden layer indexes from 0, i.e. layer == 0 means the first hidden layer.
	def preprocess(self, layer, node):
		n_layers = len(self.hidden_layers)
		while layer < 0:
			layer += n_layers
		if layer >= n_layers:
			layer = n_layers - 1

		n_nodes = self.hidden_layers[layer].M
		while node < 0:
			node += n_nodes
		if node >= n_nodes:
			node = n_nodes - 1

		return layer, node

	def forward_layer(self, X, layer):
		Z = X
		for i in range(layer + 1):
			Z = self.hidden_layers[i].forward(Z)
		return Z

	# we assume hidden layer indexes from 0, i.e. layer == 0 means the first hidden layer.
	def fit_to_input(self, layer, node, epochs=2000, learning_rate=1e-5, momentum=0.99, reg_l2=0, debug=False, print_period=20, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		reg = np.float32(reg_l2)
		one = np.float32(1)

		layer, node = self.preprocess(layer, node)
		X0 = init_weights((1, self.D))
		X = theano.shared(X0, 'X_shared')
		dX = theano.shared(np.zeros(X0.shape, dtype=np.float32), 'dX_shared')
		Y = self.forward_layer(X, layer)
		# t = np.zeros(self.hidden_layers[layer].M, dtype=np.float32)
		# t[node] = one

		# choose Y[0] because it's shape 1xM, we want just a M-size vector, not a 1xM matrix
		# cost = -T.mean(t*T.log(Y[0]) + (one - t)*T.log(one - Y[0])) + reg * T.mean(X * X)

		cost = -T.log(Y[0, node]) + reg * T.mean(X * X)

		updates = [
			(X, X + mu*dX - lr*T.grad(cost, X)),
			(dX, mu*dX - lr*T.grad(cost, X))
		]

		train_op = theano.function(
			inputs=[],
			outputs=[cost, Y],
			updates=updates
		)

		costs = []
		bestX = None
		for i in range(epochs):
			the_cost, output = train_op()
			costs.append(the_cost)

			if debug:
				if i == 0:
					print('output.shape:', output.shape)
				if i % print_period == 0:
					print('epoch=%d, cost=%.8f' % (i, the_cost))

			if the_cost > costs[-1] or np.isnan(the_cost):
				if debug:
					print('Early break at epoch=%d' % i)
				break

			bestX = X.get_value()

		if debug and show_fig:
			plt.plot(costs)
			plt.title('Costs')
			plt.show()

		return bestX

	def save(self, filename):
		arrays = [p.get_value() for layer in self.hidden_layers for p in layer.params]
		np.savez(filename, *arrays)

	@staticmethod
	def load(filename, UnsupervisedModel=AutoEncoder, activation_type=1, cost_type=1):
		dbn = DBN([], UnsupervisedModel)
		npz = np.load(filename)
		dbn.hidden_layers = []
		count = 0
		for i in range(0, len(npz.files), 3):
			W = npz['arr_%s' % i]
			bh = npz['arr_%s' % (i + 1)]
			bo = npz['arr_%s' % (i + 2)]

			if i == 0:
				dbn.D = W.shape[0]

			layer = UnsupervisedModel.createFromArray(W, bh, bo, count, activation_type, cost_type)
			dbn.hidden_layers.append(layer)
			count += 1
		return dbn


# a small demo
def main(Ntrain=-1000, sample_size=1000, hidden_layer_sizes=[1000,750,500], UnsupervisedModel=AutoEncoder, pretrain_epochs=3, pretrain_lr=0.5, pretrain_mu=0.99, debug=False):
	from util import get_mnist
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA

	Xtrain, Ytrain, _, _ = get_mnist(Ntrain)

	dbn = DBN(hidden_layer_sizes, UnsupervisedModel)
	output = dbn.fit(Xtrain, pretrain_epochs=pretrain_epochs, pretrain_lr=pretrain_lr, pretrain_mu=pretrain_mu, debug=debug)
	print('\nAfter pretrained output.shape:', output.shape)

	# sampling before using t-SNE because t-SNE requires lots of RAM
	tsne = TSNE()
	reduced = tsne.fit_transform(output[:sample_size])
	plt.scatter(reduced[:,0], reduced[:,1], c=Ytrain[:sample_size], s=100, alpha=0.5)
	plt.title('t-SNE Visualization after Pretrained')
	plt.show()

	# t-SNE on raw data
	reduced = tsne.fit_transform(Xtrain[:sample_size])
	plt.scatter(reduced[:,0], reduced[:,1], c=Ytrain[:sample_size], s=100, alpha=0.5)
	plt.title('t-SNE Visualization without Pretrained')
	plt.show()

	# PCA on pretrained data
	pca = PCA()
	reduced = pca.fit_transform(output[:sample_size])
	plt.scatter(reduced[:,0], reduced[:,1], c=Ytrain[:sample_size], s=100, alpha=0.5)
	plt.title('PCA Visualization after Pretrained')
	plt.show()

	# PCA on raw data
	reduced = pca.fit_transform(Xtrain[:sample_size])
	plt.scatter(reduced[:,0], reduced[:,1], c=Ytrain[:sample_size], s=100, alpha=0.5)
	plt.title('PCA Visualization without Pretrained')
	plt.show()


if __name__ == '__main__':
	# main(21000, 1000, [1000, 750, 500], AutoEncoder, pretrain_lr=0.5, pretrain_mu=0.99, pretrain_epochs=3, debug=False)
	main(21000, 1000, [1000, 750, 500], RBM, pretrain_lr=0.5, pretrain_mu=0, pretrain_epochs=3, debug=False)

