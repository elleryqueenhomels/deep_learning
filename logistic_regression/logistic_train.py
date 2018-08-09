import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forward(X, W, b):
	return sigmoid(X.dot(W) + b)

# T means targets/real values; pY means predictions/outputs of logistic.
def classification_rate(T, pY):
	return np.mean(T == pY)

def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

# get the data & shuffle them to be in order
X, Y = get_binary_data()
X, Y = shuffle(X, Y)

# create train and test sets
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

# randomly initialize weights
D = X.shape[1]
W = np.random.randn(D)
b = 0 # bias term

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
	pYtrain = forward(Xtrain, W, b)
	pYtest = forward(Xtest, W, b)

	ctrain = cross_entropy(Ytrain, pYtrain)
	ctest = cross_entropy(Ytest, pYtest)
	train_costs.append(ctrain)
	test_costs.append(ctest)

	# use Gradient Descent to optimize the weights & bias term
	W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)
	b -= learning_rate * (pYtrain - Ytrain).sum()

	if i % 1000 == 0:
		rtrain = classification_rate(Ytrain, np.round(pYtrain))
		rtest = classification_rate(Ytest, np.round(pYtest))
		print(i, ctrain, ctest, rtrain, rtest)

print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classificationo_rate:", classification_rate(Ytest, np.round(pYtest)))

# plt.plot(train_costs, label = 'train cost')
# plt.plot(test_costs, label = 'test cost')
# plt.legend()
# plt.show()

legend1, = plt.plot(train_costs, label = 'train cost')
legend2, = plt.plot(test_costs, label = 'test cost')
plt.legend([legend1, legend2])
plt.show()