import numpy as np
import pickle
from ann import ANN
from util import get_data

MODEL_PATH = 'C:\\Users\\Administrator\\Desktop\\python\\ann_model.pkl'

def experiment1():
	import matplotlib.pyplot as plt

	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes

	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2, 2])
	X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

	# plt.scatter(X[:,0], X[:,1], c=Y, s=50, alpha=0.5)
	# plt.show()

	model = ANN(layers=[5,6,7,6,5], activation_type=1)
	model.fit(X, Y, learning_rate=10e-5, epochs=10000)
	print('Final score:', model.score(X, Y))

# experiment on MNIST dataset.
def experiment2():
	X, Y = get_data()
	print('\nX.shape =', X.shape, '; Y.shape =', Y.shape, '\n')

	Ntrain = int(len(Y) / 4)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	# For MNIST Dataset the best hyperparameters: layers=[50], activation_type=2, learning_rate=10e-5, epochs=2000
	model = ANN(layers=[30,30,30,30], activation_type=1)
	model.fit(Xtrain, Ytrain, learning_rate=10e-5, epochs=10000)
	with open(MODEL_PATH, 'wb') as f:
		pickle.dump(model, f)
	print('Final score:', model.score(Xtest, Ytest))


if __name__ == '__main__':
	experiment2()
