import numpy as np
import matplotlib.pyplot as plt

N = 50

# generate the data
X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn(N)

# make outliers
Y[-1] += 30
Y[-2] += 30

# scatter the data
plt.scatter(X, Y)
plt.show()

# add bias term
X = np.vstack([np.ones(N), X]).T

# plot the Maximum Likelihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) # ml is short for Maximum Likelihood
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)
plt.show()

# plot the L2 Regularization solution
l2 = 1000.0 # l2 is the lambda
w_map = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y)) # map is short for Maximize A Posterior
Yhat_map = X.dot(w_map)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label = 'maximum likelihood')
plt.plot(X[:, 1], Yhat_map, label = 'map')
plt.legend()
plt.show()