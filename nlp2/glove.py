# GLoVe using Gradient Descent or Alternating Least Squares

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from word2vec import find_analogies
from util import get_wikipedia_data
from util import get_sentences_with_word2idx_limit_vocab


# using ALS, what's the least # files to get correct analogies?
# use this for word2vec training to make it faster
# first tried 20 files --> not enough
# how about 30 files --> some correct but still not enough
# 40 files --> half right but 50 is better


class Glove:

    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, use_gd=False):
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        # TODO: would it better to use a sparse matrix?
        t0 = datetime.now()

        V = self.V
        D = self.D

        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print('Number of sentences to process: %d' % N)

            for it, sentence in enumerate(sentences):
                if it % 10000 == 0:
                    print('processed: %d / %d' % (it, N))

                n = len(sentence)
                for i in range(n):
                    # i is not the word index
                    # j is not the word index
                    # i just points to the i-th element of the sequence(sentence) we are looking at
                    wi = sentence[i]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure 'START' and 'END' tokens are part of some context
                    # otherwise their f(X) will be 0 (denominator in bias update)
                    # 'START' word index is 0, 'END' word index is 1
                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi,0] += points
                        X[0,wi] += points
                    if i + self.context_sz > n:
                        points = 1.0 / (n - i)
                        X[wi,1] += points
                        X[1,wi] += points

                    # left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j)
                        X[wi,wj] += points
                        X[wj,wi] += points

                    # right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i)
                        X[wi,wj] += points
                        X[wj,wi] += points

            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        # weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax))**alpha
        fX[X >= xmax] = 1

        # target
        logX = np.log(X + 1)

        print('max in X: %s' % X.max())
        print('max in f(X): %s' % fX.max())
        print('max in log(X): %s' % logX.max())
        print('>>> Elapsed time to build co-occurrence matrix: %s' % (datetime.now() - t0))

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        costs = []
        sentence_indexes = np.arange(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = (fX * delta * delta).sum()
            costs.append(cost)
            print('epoch: %d, cost: %f' % (epoch, cost))

            if use_gd:
                # gradient descent method

                # update W
                # oldW = W.copy()
                for i in range(V):
                    # for j in range(V):
                    #     W[i] -= learning_rate * fX[i,j] * (W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j]) * U[j]
                    W[i] -= learning_rate * (fX[i] * delta[i]).dot(U)
                W -= learning_rate * reg * W

                # update b
                for i in range(V):
                    # for j in range(V):
                    #     b[i] -= learning_rate * fX[i,j] * (W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j])
                    b[i] -= learning_rate * fX[i].dot(delta[i])
                # b -= learning_rate * reg * b

                # update U
                for j in range(V):
                    # for i in range(V):
                    #     U[j] -= learning_rate * fX[i,j] * (W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j]) * W[i]
                    U[j] -= learning_rate * (fX[:,j] * delta[:,j]).dot(W)
                U -= learning_rate * reg * U

                # update c
                for j in range(V):
                    # for i in range(V):
                    #     c[j] -= learning_rate * fX[i,j] * (W[i].dot(U[j] + b[i] + c[j] + mu - logX[i,j]))
                    c[j] -= learning_rate * fX[:,j].dot(delta[:,j])
                # c -= learning_rate * reg * c

            else:
                # ALS method

                # update W
                # slow way
                # for i in range(V):
                #     matrix = reg * np.eye(D)
                #     vector = 0
                #     for j in range(V):
                #         matrix += fX[i,j] * np.outer(U[j], U[j])
                #         vector += fX[i,j] * (logX[i,j] - b[i] - c[j] - mu) * U[j]
                #     W[i] = np.linalg.solve(matrix, vector)
                
                # update W
                # fast way
                for i in range(V):
                    matrix = reg * np.eye(D) + (fX[i] * U.T).dot(U)
                    vector = (fX[i] * (logX[i] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)

                # update b
                for i in range(V):
                    denominator = fX[i].sum()
                    numerator = fX[i].dot(logX[i] - W[i].dot(U.T) - c - mu)
                    b[i] = numerator / denominator / (1 + reg)

                # update U
                for j in range(V):
                    matrix = reg * np.eye(D) + (fX[:,j] * W.T).dot(W)
                    vector = (fX[:,j] * (logX[:,j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)

                # update c
                for j in range(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W.dot(U[j]) - b - mu)
                    c[j] = numerator / denominator / (1 + reg)

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, filename):
        # function word_analogies expects a (V,D) matrix and a (D,V) matrix
        arrays = [self.W, self.U.T]
        np.savez(filename, *arrays)


def main(we_file, w2i_file, use_brown=True, n_files=50):
    if use_brown:
        cc_matrix = 'cc_matrix_brown.npy'
    else:
        cc_matrix = 'cc_matrix_%s.npy' % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = [] # dummy - we don't actually use it
    else:
        if use_brown:
            keep_words = set([
                'king', 'man', 'woman',
                'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
                'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
                'australia', 'australian', 'december', 'november', 'june',
                'january', 'february', 'march', 'april', 'may', 'july', 'august',
                'september', 'october',
            ])
            sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)

        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    D = 100
    V = len(word2idx)
    context_sz = 10
    print('D = %d, V = %d' % (D, V))
    model = Glove(D, V, context_sz)

    # alternating least squares method
    model.fit(sentences, cc_matrix=cc_matrix, epochs=20, use_gd=False)

    # gradient descent method
    # model.fit(
    #     sentences,
    #     cc_matrix=cc_matrix,
    #     learning_rate=1e-4,
    #     reg=0.1,
    #     epochs=500,
    #     use_gd=True,
    # )

    model.save(we_file)


if __name__ == '__main__':
    use_brown = True
    n_files = 50

    if use_brown:
        we = 'glove_model_brown.npz'
        w2i = 'glove_word2idx_brown.json'
    else:
        we = 'glove_model_%s.npz' % n_files
        w2i = 'glove_word2idx_%s.json' % n_files

    main(we, w2i, use_brown=True, n_files=n_files)

    w1 = ['king', 'france', 'france', 'paris', 'france', 'japan', 'japan', 'japan', 'december']
    w2 = ['man', 'paris', 'paris', 'france', 'french', 'japanese', 'japanese', 'japanese', 'november']
    w3 = ['woman', 'london', 'rome', 'italy', 'english', 'chinese', 'italian', 'australian', 'june']
    for use_concat in (True, False):
        print('** use_concat: %s' % use_concat)
        find_analogies(w1, w2, w3, use_concat, we_file=we, w2i_file=w2i)

