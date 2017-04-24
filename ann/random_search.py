# Random Search to choose hyperparameters

import numpy as np

from sklearn.utils import shuffle
from util import get_spiral, get_clouds
from ann_theano import ANN


def random_search():
	# get the data and split into train/test set
	X, Y = get_spiral()
	# X, Y = get_clouds()

	X, Y = shuffle(X, Y)
	Ntrain = int(0.7 * len(X))
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# hyperparameters settings
	M = 20
	n_hiddens = 2
	log_lr = -4
	log_l2 = -2
	max_tries = 30

	# loop through all possible hyperparameters settings
	best_validation_rate = 0
	best_M = None
	best_n_hiddens = None
	best_lr = None
	best_l2 = None
	for _ in range(max_tries):
		model = ANN([M]*n_hiddens)
		model.fit(Xtrain, Ytrain, learning_rate=10**log_lr, reg_l2=10**log_l2, momentum=0.99, epochs=3000)
		validation_accuracy = model.score(Xtest, Ytest)
		train_accuracy = model.score(Xtrain, Ytrain)
		print(
			'validation_accuracy: %.3f%%, train_accuracy: %.3f%%, settings: %s, %s, %s' % 
			(validation_accuracy*100, train_accuracy*100, [M]*n_hiddens, log_lr, log_l2)
		)
		if validation_accuracy > best_validation_rate:
			best_validation_rate = validation_accuracy
			best_M = M
			best_n_hiddens = n_hiddens
			best_lr = log_lr
			best_l2 = log_l2

		# randomly choose new hyperparameters
		n_hiddens = best_n_hiddens + np.random.randint(-1, 2) # -1 or 0 or 1
		n_hiddens = max(1, n_hiddens)
		M = best_M + np.random.randint(-1, 2) * 10
		M = max(10, M)
		log_lr = best_lr + np.random.randint(-1, 2)
		log_l2 = best_l2 + np.random.randint(-1, 2)
	print('Best validation accuracy: %.6f%%' % (best_validation_rate*100))
	print('Best settings:')
	print('hidden_layer_sizes: %s' % ([best_M]*best_n_hiddens))
	print('learning_rate: %s' % best_lr)
	print('l2_reg: %s' % best_l2)


if __name__ == '__main__':
	random_search()

