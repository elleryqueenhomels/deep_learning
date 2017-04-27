import numpy as np

from dnn import DNN
from cnn import CNN
from rbm import RBM
from autoencoder import AutoEncoder
from util import get_mnist


def test_autoencoder(Xtrain, Ytrain, Xtest, Ytest):
	dnn = DNN([1000, 750, 500], UnsupervisedModel=AutoEncoder, activation_type=1, cost_type=1)

	# args: (Xtest, Ytest, epochs=1, batch_sz=100, pretrain=True, pretrain_epochs=1, pretrain_batch_sz=100, learning_rate=0.01, momentum=0.99, debug=False, print_period=20, show_fig=False)
	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3, batch_sz=100, debug=True, show_fig=False)

	print('\nFinal Score:')
	print('Train accuracy=%.6f%%, Train size: %s' % (dnn.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Test accuracy=%.6f%%, Test size: %s' % (dnn.score(Xtest, Ytest)*100, len(Ytest)))


def test_rbm(Xtrain, Ytrain, Xtest, Ytest):
	dnn = DNN([1000, 750, 500], UnsupervisedModel=RBM, activation_type=1, cost_type=1)

	# args: (Xtest, Ytest, epochs=1, batch_sz=100, pretrain=True, pretrain_epochs=1, pretrain_batch_sz=100, learning_rate=0.01, momentum=0.99, debug=False, print_period=20, show_fig=False)
	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3, batch_sz=100, debug=True, show_fig=False)

	print('\nFinal Score:')
	print('Train accuracy=%.6f%%, Train size: %s' % (dnn.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Test accuracy=%.6f%%, Test size: %s' % (dnn.score(Xtest, Ytest)*100, len(Ytest)))


def test_no_pretrain(Xtrain, Ytrain, Xtest, Ytest):
	dnn = DNN([1000, 750, 500], UnsupervisedModel=RBM, activation_type=1, cost_type=1)

	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=False, epochs=3, batch_sz=100, debug=True, show_fig=False)

	print('\nFinal Score:')
	print('Train accuracy=%.6f%%, Train size: %s' % (dnn.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Test accuracy=%.6f%%, Test size: %s' % (dnn.score(Xtest, Ytest)*100, len(Ytest)))


def test_cnn(Xtrain, Ytrain, Xtest, Ytest):
	N, D = Xtrain.shape
	d = int(np.sqrt(D))
	Xtrain = Xtrain.reshape(N, 1, d, d)
	Xtest = Xtest.reshape(len(Xtest), 1, d, d)

	cnn = CNN([(20, 5, 5), (50, 5, 5)], hidden_layer_sizes=[1000, 750, 500], UnsupervisedModel=RBM, activation_type=1, cost_type=1)

	cnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3, batch_sz=100, debug=True, show_fig=False)

	print('\nFinal Score:')
	print('Train accuracy=%.6f%%, Train size: %s' % (cnn.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Test accuracy=%.6f%%, Test size: %s' % (cnn.score(Xtest, Ytest)*100, len(Ytest)))


def main():
	Xtrain, Ytrain, Xtest, Ytest = get_mnist(21000)
	test_autoencoder(Xtrain, Ytrain, Xtest, Ytest)
	test_rbm(Xtrain, Ytrain, Xtest, Ytest)
	test_no_pretrain(Xtrain, Ytrain, Xtest, Ytest)
	test_cnn(Xtrain, Ytrain, Xtest, Ytest)


if __name__ == '__main__':
	main()

