import numpy as np
from process import get_binary_data

X, Y = get_binary_data()

D = X.shape[1] # (N, D) = (# of Samples, # of Features)
W = np.random.randn(D)
b = 0

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forward(X, W, b):
	return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

# calculate the accuracy
def classification_rate(Y, P):
	return np.mean(Y == P) # It looks like Y == P will return array of True and False, but it will return array of ones and zeros

print("Accuracy:", classification_rate(Y, predictions))