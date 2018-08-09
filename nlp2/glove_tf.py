# GLoVe using Gradient Descent in TensorFlow

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from word2vec import find_analogies
from util import get_wikipedia_data
from util import get_sentences_with_word2idx_limit_vocab


def momentum_updates(cost, params, learning_rate=1e-4, mu=0.9):
    lr = np.float32(learning_rate)
    mu = np.float32(mu)
    grads = T.grad(cost, params)
    velocities = [theano.shared(
        np.zeros_like(p.get_value()).astype(np.float32)) for p in params]
    updates = []
    for p, v, g in zip(params, velocities, grads):
        new_v = mu * v - lr * g
        new_p = p + new_v
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates


class Glove:

    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, mu=0.9, xmax=100, alpha=0.75, epochs=10):
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        # TODO: would it be better to use a sparse matrix?
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

        # cast
        fX = fX.astype(np.float32)
        logX = logX.astype(np.float32)

        print('max in X: %s' % X.max())
        print('max in f(X): %s' % fX.max())
        print('max in log(X): %s' % logX.max())
        print('>>> Elapsed time to build co-occurrence matrix: %s' % (datetime.now() - t0))

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V).reshape(V, 1)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V).reshape(1, V)
        mu = logX.mean()

        # initialize weights, inputs, targets placeholders
        tfW = tf.Variable(W.astype(np.float32))
        tfb = tf.Variable(b.astype(np.float32))
        tfU = tf.Variable(U.astype(np.float32))
        tfc = tf.Variable(c.astype(np.float32))
        tffX = tf.placeholder(tf.float32, shape=(V, V), name='fX')
        tflogX = tf.placeholder(tf.float32, shape=(V, V), name='logX')

        delta = tf.matmul(tfW, tf.transpose(tfU)) + tfb + tfc + mu - tflogX
        cost = tf.reduce_sum(tffX * delta * delta)

        # regularization
        regularized_cost = cost
        for param in (tfW, tfb, tfW, tfc):
            regularized_cost += reg * tf.reduce_sum(param * param)

        train_op = tf.train.MomentumOptimizer(
            learning_rate, 
            momentum=mu
        ).minimize(regularized_cost)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(regularized_cost)

        costs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sentence_indexes = np.arange(len(sentences))
            for epoch in range(epochs):
                c, _ = sess.run((cost, train_op), 
                    feed_dict={tflogX: logX, tffX: fX})
                print('epoch: %d, cost: %f' % (epoch, c))
                costs.append(c)

            self.W, self.U = sess.run([tfW, tfU])

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
    # model.fit(sentences, cc_matrix=cc_matrix, epochs=20, use_gd=False)

    # gradient descent method
    model.fit(
        sentences,
        cc_matrix=cc_matrix,
        learning_rate=1e-4,
        reg=0.1,
        epochs=500,
    )

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

