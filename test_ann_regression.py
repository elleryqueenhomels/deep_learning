import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ann_regression import ANN_Regression


def experiment1():
	X = np.linspace(-20, 20, 2000)
	Y = np.sin(X) / X

	plt.plot(X, Y)
	plt.show()

	# Xmax = np.max(np.abs(X))
	# Xnorm = X / Xmax
	Xnorm = (X - X.mean()) / X.std()

	model = ANN_Regression([100], activation_type=1)
	# if overflow in fit() function, use learning_rate=10e-6 or less.
	model.fit(Xnorm, Y, learning_rate=10e-5, epochs=20000)
	print('\nFinal score: %.8f%%' % (model.score(Xnorm, Y) * 100), '\n')

	Yhat = model.predict(Xnorm)
	plt.plot(X, Y, label='target')
	plt.plot(X, Yhat, label='prediction')
	plt.legend()
	plt.show()


def experiment2():
	df = pd.read_excel('mlr02.xls')
	X = df.as_matrix()

	# using age to predict systolic blood pressure
	plt.scatter(X[:,1], X[:,0])
	plt.show()
	# looks pretty linear!

	# using weight to predict systolic blood pressure
	plt.scatter(X[:,2], X[:,0])
	plt.show()
	# looks pretty linear!


if __name__ == '__main__':
	experiment1()