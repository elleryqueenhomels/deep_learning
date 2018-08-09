import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import get_data, getFacialData
from bayes_classifier import Bayes
from bayes_optimal import Bayes_OPT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from ann import ANN

training_mode = True
isMNIST = True
save_mode = False
facial = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_PATH = '/Users/WenGao_Ye/Desktop/python_test/supervised_machine_learning/mymodel.pkl'

def y2indicator(Y):
	N = len(Y)
	K = len(set(Y))

	T = np.zeros((N, K))
	for i in range(N):
		T[i, int(Y[i])] = 1
	return T

if __name__ == '__main__':
	if training_mode:
		print('\nBegin to extract data.')
		t0 = datetime.now()
		if isMNIST:
			X, Y = get_data()
		else:
			X, Y = getFacialData(balance_ones=True)
		print('\nFinish extracting data. Time:', (datetime.now() - t0))

		Ntrain = int(len(Y) / 4)
		Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]

		# model = ANN([50, 50], activation_type=2)
		model = Bayes()
		# model = Bayes_OPT()
		# model = RandomForestClassifier()
		# model = Perceptron()

		print('\nBegin to training model.')
		t0 = datetime.now()
		model.fit(Xtrain, Ytrain)
		print('\nTraining time:', (datetime.now() - t0), '; Train size:', len(Ytrain))

		# just in case you are curious
		Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
		print('\nBegin to testing model.')
		t0 = datetime.now()
		print('\nTest accuracy:', model.score(Xtest, Ytest))
		print('\nTest time:', (datetime.now() - t0), '; Test size:', len(Ytest))

		if not save_mode:
			exit()

		with open(MODEL_PATH, 'wb') as f:
			pickle.dump(model, f)

		print('\nSave model successfully!')
		print('Model type: %s\n' % type(model))
	else:
		with open(MODEL_PATH, 'rb') as f:
			model = pickle.load(f)

		print('\nModel type:', type(model))

		print('\nBegin to extract data.')
		t0 = datetime.now()
		if isMNIST:
			X, Y = get_data()
		else:
			X, Y = getFacialData(balance_ones=False)
		print('\nFinish extracting data. Time:', (datetime.now() - t0))

		N, D = X.shape
		while True:
			i = np.random.choice(N)
			y = model.predict(X[i].reshape(1, D))[0]
			if isMNIST:
				print('\nprediction:', y)
				print('target:', Y[i])
				plt.imshow(X[i].reshape(28, 28))
				plt.show()
			else:
				print('\nprediction:', facial[y])
				print('target:', facial[Y[i]])
				plt.imshow(X[i].reshape(48, 48))
				plt.show()

			response = input('\nContinue? (Y/N)\n')
			if response in ('n', 'N'):
				break