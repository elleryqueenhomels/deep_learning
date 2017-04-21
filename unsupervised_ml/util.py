import numpy as np
import pandas as pd

DATA_PATH = '../data_set/MNIST_train.csv'

def get_mnist():
	data = pd.read_csv(DATA_PATH).as_matrix().astype(np.float32)
	np.random.shuffle(data)

	Xtrain = data[:-1000, 1:] / np.float32(255)
	Ytrain = data[:-1000, 0].astype(np.int32)
	Xtest = data[-1000:, 1:] / np.float32(255)
	Ytest = data[-1000:, 1:].astype(np.int32)

	return Xtrain, Ytrain, Xtest, Ytest

