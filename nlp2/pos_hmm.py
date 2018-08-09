# Hidden Markov Model for predicting POS tags

import numpy as np
import matplotlib.pyplot as plt

from hmm_scaled import HMM
from pos_baseline import get_data

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def predict(X, hmm):
    P = []
    for x in X:
        p = hmm.get_state_sequence(x)
        P.append(p)
    return P


def main(smoothing=1e-1):
    # X = words, Y = POS tags
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1

    # find hidden state transition matrix and pi
    M = max(max(y) for y in Ytrain) + 1 # len(set(flatten(Ytrain)))
    A = np.ones((M, M)) * smoothing # add-one smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        pi[y[0]] += 1
        for i in range(len(y) - 1):
            A[y[i], y[i + 1]] += 1
    # turn it into a probability matrix
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    # find the observation matrix
    B = np.ones((M, V)) * smoothing # add-one smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis=1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # get predictions
    Ptrain = predict(Xtrain, hmm)
    Ptest = predict(Xtest, hmm)

    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))


if __name__ == '__main__':
    main()

