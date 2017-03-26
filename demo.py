import numpy as np
import pickle
from ann import ANN
from util import get_data, get_facial_data, get_xor, get_donut
from datetime import datetime

TRAIN_MODE = True
MODEL_PATH = '/Users/WenGao_Ye/Desktop/deep_learning/ann_model_facial.pkl'

def experiment1():
	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes

	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2, 2])
	X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	
	# import matplotlib.pyplot as plt
	# plt.scatter(X[:,0], X[:,1], c=Y, s=50, alpha=0.5)
	# plt.show()

	model = ANN(layers=[5,6,7,6,5], activation_type=1)
	model.fit(X, Y, learning_rate=10e-5, epochs=10000)
	print('\nFinal score: %.8f%%' % (model.score(X, Y) * 100))

# experiment on MNIST dataset.
def experiment2():
	print('\nBegin to extract data.')
	t0 = datetime.now()
	X, Y = get_data()
	# X, Y = get_facial_data()
	print('\nFinish extracting data. Time:', (datetime.now() - t0))
	print('\nX.shape =', X.shape, '; Y.shape =', Y.shape)

	Ntrain = int(len(Y) / 4)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	if TRAIN_MODE:
		# For MNIST Dataset the best hyperparameters: layers=[100], activation_type=1, learning_rate=10e-5, epochs=5000
		model = ANN(layers=[100], activation_type=1)
		print('\nBegin to training model.')
		t0 = datetime.now()
		model.fit(Xtrain, Ytrain, learning_rate=10e-5, epochs=5000)
		print('\nTraining time:', (datetime.now() - t0), 'Train size:', len(Ytrain))

		with open(MODEL_PATH, 'wb') as f:
			pickle.dump(model, f)
		print('\nSave model successfully! Model type:', type(model))
	else:
		with open(MODEL_PATH, 'rb') as f:
			model = pickle.load(f)
	
	print('\nBegin to testing model.')
	t0 = datetime.now()
	print('\nTest accuracy: %.8f%%' % (model.score(Xtest, Ytest) * 100))
	print('Test time:', (datetime.now() - t0), 'Test size:', len(Ytest), '\n')

def experiment3():
	X, Y = get_xor()
	X, Y = shuffle(X, Y)

	# import matplotlib.pyplot as plt
	# plt.scatter(X[:,0], X[:,1], c=Y, s=50, alpha=0.5)
	# plt.show()

	model = ANN([10], activation_type=1) # 5 units or more
	model.fit(X, Y, learning_rate=10e-5, epochs=100000)
	print('\nIn XOR: final score = %.8f%%' % (model.score(X, Y) * 100), '\n')

def experiment4():
	X, Y = get_donut()
	X, Y = shuffle(X, Y)

	# import matplotlib.pyplot as plt
	# plt.scatter(X[:,0], X[:,1], c=Y, s=50, alpha=0.5)
	# plt.show()

	model = ANN([8], activation_type=1) # 8 units or more
	model.fit(X, Y, learning_rate=10e-5, epochs=100000)
	print('\nIn Donut: final score = %.8f%%' % (model.score(X, Y) * 100), '\n')


if __name__ == '__main__':
	experiment2()
