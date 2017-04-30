# Visualize what features a neuron has learned

import numpy as np
import matplotlib.pyplot as plt

from util import get_mnist
from dbn import DBN
from rbm import RBM
from autoencoder import AutoEncoder


def main(Ntrain=-1000, layer=-1, loadfile=None, savefile=None, hidden_layer_sizes=[1000,750,500], UnsupervisedModel=AutoEncoder, pretrain_epochs=15, pretrain_lr=0.5, pretrain_mu=0.99, debug=False):
	Xtrain, Ytrain, _, _ = get_mnist(Ntrain)

	if loadfile is not None:
		dbn = DBN.load(loadfile, UnsupervisedModel)
	else:
		dbn = DBN(hidden_layer_sizes, UnsupervisedModel)
		dbn.fit(Xtrain, pretrain_epochs=pretrain_epochs, pretrain_lr=pretrain_lr, pretrain_mu=pretrain_mu, debug=debug)

	if savefile is not None:
		dbn.save(savefile)

	# first layer features, just print out the weight and reshape(28, 28) is the feature
	# initial weight is D x M
	W = dbn.hidden_layers[0].W.get_value()
	# for i in range(W.shape[1]):
	while True:
		i = np.random.choice(W.shape[1])
		plt.imshow(W[:,i].reshape(28,28), cmap='gray')
		plt.title('Layer 0, Node %d' % i)
		plt.show()
		should_quit = input("Show more? Enter 'n' or 'N' to quit...\n")
		if should_quit in ('n', 'N'):
			break

	# features learned in specific layer
	layer, _ = dbn.preprocess(layer, 0)
	n_nodes = dbn.hidden_layers[layer].M
	for node in range(n_nodes):
	# while True:
		# node = np.random.choice(n_nodes)
		# activate the node
		X = dbn.fit_to_input(layer, node, epochs=2000, learning_rate=1e-5, momentum=0.99, reg_l2=0, debug=True, print_period=50, show_fig=True)
		# X = dbn.fit_to_input(layer, node, epochs=20000, learning_rate=1e-2, momentum=0.99, reg_l2=0.01, debug=True, print_period=1000, show_fig=True)
		plt.imshow(X.reshape(28,28), cmap='gray')
		plt.title('Layer %d, Node %d' % (layer, node))
		plt.show()
		should_quit = input("Show more? Enter 'n' or 'N' to quit...\n")
		if should_quit in ('n', 'N'):
			break


if __name__ == '__main__':
	# to load a saved file
	# main(loadfile='ae15.npz', layer=-1)
	main(loadfile='rbm15.npz', layer=-1)

	# to neither load or save
	# main(Ntrain=21000, layer=-1, hidden_layer_sizes=[500,500,10], UnsupervisedModel=AutoEncoder, pretrain_epochs=15, pretrain_lr=0.5, pretrain_mu=0.99, debug=False)
	# main(Ntrain=21000, layer=-1, hidden_layer_sizes=[500,500,10], UnsupervisedModel=RBM, pretrain_epochs=15, pretrain_lr=0.5, pretrain_mu=0, debug=False)

	# to save a trained unsupervised deep network
	# main(Ntrain=21000, layer=-1, savefile='ae15.npz', hidden_layer_sizes=[500,500,10], UnsupervisedModel=AutoEncoder, pretrain_epochs=15, pretrain_lr=0.5, pretrain_mu=0.99, debug=False)
	# main(Ntrain=21000, layer=-1, savefile='rbm15.npz', hidden_layer_sizes=[500,500,10], UnsupervisedModel=RBM, pretrain_epochs=15, pretrain_lr=0.5, pretrain_mu=0, debug=False)


