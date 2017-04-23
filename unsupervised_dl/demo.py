import numpy as np

from autoencoder import DNN
from util import get_mnist

def main():
	Xtrain, Ytrain, Xtest, Ytest = get_mnist(21000)

	dnn = DNN([1000, 750, 500], activation_type=1, cost_type=1)

	# args: (Xtest, Ytest, epochs=1, batch_sz=100, pretrain=True, pretrain_epochs=1, pretrain_batch_sz=100, learning_rate=0.01, momentum=0.99, debug=False, print_period=20, show_fig=False)
	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3, batch_sz=100, debug=True, show_fig=False)
	# vs.
	# dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=10, batch_sz=100, pretrain=False, debug=True, show_fig=False)

	print('\nFinal Score:')
	print('Train accuracy=%.6f%%, Train size: %s' % (dnn.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Test accuracy=%.6f%%, Test size: %s' % (dnn.score(Xtest, Ytest)*100, len(Ytest)))


if __name__ == '__main__':
	main()
