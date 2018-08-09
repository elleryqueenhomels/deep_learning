import pickle
import numpy as np
from util import get_data
from sklearn.ensemble import RandomForestClassifier
from bayes_classifier import Bayes

if __name__ == '__main__':
	X, Y = get_data()
	Ntrain = int(len(Y) / 4)
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]

	# model = RandomForestClassifier()
	model = Bayes()
	model.fit(Xtrain, Ytrain)

	# just in case you are curious
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
	print('\nTest accuracy:', model.score(Xtest, Ytest), '\n')

	with open('mymodel.pkl', 'wb') as f:
		pickle.dump(model, f)