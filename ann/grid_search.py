# Grid Search to choose hyperparameters

import numpy as np

from sklearn.utils import shuffle
from util import get_spiral, get_clouds
from ann_theano import ANN


def grid_search():
	# get the data and split into train/test set
	X, Y = get_spiral()
	# X, Y = get_clouds()

	X, Y = shuffle(X, Y)
	Ntrain = int(0.7 * len(X))
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# hyperparameters to try
	hidden_layer_sizes = [
		[300],
		[100,100],
		[50,50,50]
	]
	learning_rates = [1e-2, 1e-3, 1e-4]
	l2_regs = [1.0, 0.0, 0.1]

	# loop through all possible hyperparameters settings
	best_validation_rate = 0
	best_hls = None
	best_lr = None
	best_l2 = None
	for hls in hidden_layer_sizes:
		for lr in learning_rates:
			for l2 in l2_regs:
				model = ANN(hls)
				model.fit(Xtrain, Ytrain, learning_rate=lr, reg_l2=l2, momentum=0.99, epochs=3000)
				validation_accuracy = model.score(Xtest, Ytest)
				train_accuracy = model.score(Xtrain, Ytrain)
				print(
					'validation_accuracy: %.3f%%, train_accuracy: %.3f%%, settings: %s, %s, %s' % 
					(validation_accuracy*100, train_accuracy*100, hls, lr, l2)
				)
				if validation_accuracy > best_validation_rate:
					best_validation_rate = validation_accuracy
					best_hls = hls
					best_lr = lr
					best_l2 = l2
	print('Best validation accuracy: %.6f%%' % (best_validation_rate*100))
	print('Best settings:')
	print('hidden_layer_sizes: %s' % best_hls)
	print('learning_rate: %s' % best_lr)
	print('l2_reg: %s' % best_l2)


if __name__ == '__main__':
	grid_search()

