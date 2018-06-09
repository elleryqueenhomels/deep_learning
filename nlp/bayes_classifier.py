# This is an example of Bayes Classifier (Non-Naive Bayes) on MNIST Dataset.

import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


class Bayes(object):

    # P(C|X) ~ P(X|C) * P(C)
    def fit(self, X, Y, smoothing=1e-3):
        N, D = X.shape
        Y = Y.astype(np.int32)
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y) # classes
        for c in labels:
            Xc = X[Y == c]
            self.gaussians[c] = {
                'mean': Xc.mean(axis=0),
                'cov': np.cov(Xc.T) + np.eye(D) * smoothing # covariance matrix: (DxD)
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        Y = Y.astype(np.int32)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.priors)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    from util import get_data
    from util import getFacialData
    
    X, Y = get_data()
    # X, Y = getFacialData(balance_ones=False)

    N = X.shape[0]
    Ntrain = int(N / 2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print('\nTraining time:', (datetime.now() - t0))

    t0 = datetime.now()
    print('\nTrain accuracy:', model.score(Xtrain, Ytrain))
    print('Train accuracy time:', (datetime.now() - t0), ' Train size:', len(Ytrain))

    t0 = datetime.now()
    print('\nTest accuracy:', model.score(Xtest, Ytest))
    print('Test accuracy time:', (datetime.now() - t0), ' Test size:', len(Ytest))