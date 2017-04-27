import numpy as np

from scipy.io import loadmat
from sklearn.utils import shuffle
# from util import shuffle  # If no sklearn installed, using this shuffle() instead.
from datetime import datetime

from cnn_tensorflow import CNN


def rearrange(X):
	# input is (32, 32, 3, N) from MATLAB file
	# output is (N, 32, 32, 3) for TensorFlow using
	W, H, C, N = X.shape
	out = np.zeros((N, W, H, C), dtype=np.float32)
	for i in range(N):
		for j in range(C):
			out[i, :, :, j] = X[:, :, j, i]
	return out / np.float32(255)

def main():
	# step 1: load the data, transform as needed
	train = loadmat('../../python_test/data_set/train_32x32.mat')
	test = loadmat('../../python_test/data_set/test_32x32.mat')
	print('\nSuccessfully extract data!\n')

	# Need to scale! don't leave as 0..255
	# Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
	# So flatten it and make it 0..9
	# Also need indicator matrix for cost function
	Xtrain = rearrange(train['X'])
	Ytrain = train['y'].flatten() - 1
	del train
	Ytrain = Ytrain.astype(np.int32)

	Xtest = rearrange(test['X'])
	Ytest = test['y'].flatten() - 1
	del test
	Ytest = Ytest.astype(np.int32)

	print('Xtrain.shape =', Xtrain.shape)
	print('Xtest.shape =', Xtest.shape)

	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Xtest, Ytest = shuffle(Xtest, Ytest)

	# model = CNN([(20, 5, 5), (50, 5, 5)], pool_layer_sizes=[(2, 2), (2, 2)], hidden_layer_sizes=[200, 300, 100])
	model = CNN([(20, 5, 5), (50, 5, 5)], hidden_layer_sizes=[500, 100])

	print('\nBegin to train model...\n')
	t0 = datetime.now()
	model.fit(Xtrain[:10000], Ytrain[:10000], epochs=5, batch_sz=500, learning_rate=10e-6, decay=0.99, momentum=0.9, reg_l2=0.01, debug=True, cal_train=False, debug_points=100, valid_set=[Xtest[:1000], Ytest[:1000]])
	print('\nTraing time:', (datetime.now() - t0))

	t0 = datetime.now()
	print('\nTraining accuracy: %.6f%%, Train size: %d' % (model.score(Xtrain, Ytrain)*100, len(Ytrain)))
	print('Elapsed time:', (datetime.now() - t0))

	t0 = datetime.now()
	print('\nTest accuracy: %.6f%%, Test size: %d' % (model.score(Xtest, Ytest)*100, len(Ytest)))
	print('Elapsed time:', (datetime.now() - t0))

	print('\nTest model.predict()!')
	P = model.predict(Xtest)
	print('After predict(), P.shape =', P.shape, ', type:', type(P))

	print('\nTest model.forward()!')
	pY = model.forward(Xtest)
	print('After forward(), pY.shape =', pY.shape, ', type:', type(pY))
	arg = np.argmax(pY, axis=1)

	print('\nnp.argmax(pY, axis=1) =', arg, ', type:', type(arg))
	print('P =', P, 'type:', type(P))
	print('np.mean(np.argmax(pY, axis=1) == P) =', np.mean(arg == P))
	assert(np.mean(np.argmax(pY, axis=1) == P) == 1)
	print('\nPass the assert(). model.forward() operates successfully!\n')

if __name__ == '__main__':
	main()
