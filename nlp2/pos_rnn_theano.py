import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from gru import GRU
from lstm import LSTM
from util import init_weight
from datetime import datetime
from pos_baseline import get_data
from sklearn.metrics import f1_score


class RNN(object):

    def __init__(self, V, D, K, hidden_layer_sizes):
        self.V = V
        self.D = D
        self.K = K
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=1e-4, mu=0.99, epochs=30, activation=T.nnet.relu, 
            RecurrentUnit=GRU, normalize=False, show_fig=True):
        V = self.V
        D = self.D
        K = self.K
        N = len(X)

        We = init_weight(V, D)
        self.hidden_layers = []

        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = init_weight(Mi, K)
        bo = np.zeros(K)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in reversed(self.hidden_layers):
            self.params += ru.params

        thX = T.ivector('X')
        thY = T.ivector('Y')

        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        # just for test
        testf = theano.function(
            inputs=[thX],
            outputs=py_x,
        )
        testout = testf(X[0])
        print('py_x.shape:', testout.shape)

        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        gWe = T.grad(cost, self.We)
        dWe = theano.shared(self.We.get_value() * 0)
        dWe_update = mu * dWe - learning_rate * gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)

        updates = [
            (p, p + mu * dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu * dp - learning_rate * g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        costs = []
        sequence_indexes = np.arange(N)
        n_total = sum(len(y) for y in Y)

        for i in range(epochs):
            t0 = datetime.now()
            np.random.shuffle(sequence_indexes)

            cost = 0
            n_correct = 0
            for it, j in enumerate(sequence_indexes):
                c, p = train_op(X[j], Y[j])
                cost += c
                n_correct += np.sum(p == Y[j])
                if it % 200 == 0:
                    print('epoch: %d, seq: %d/%d, correct rate so far: %.3f%%, cost so far: %.3f' % 
                        (i, it, N, n_correct / n_total * 100, cost))

            print('>>> epoch: %d, cost: %.3f, correct rate: %.3f%%, elapsed time: %s' % 
                (i, cost, n_correct/ n_total * 100, datetime.now() - t0))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.title('Costs')
            plt.show()

    def predict(self, X):
        return [self.predict_op(x) for x in X]

    def score(self, X, Y):
        n_total = sum(len(y) for y in Y)
        n_correct = 0
        for x, y in zip(X, Y):
            p = self.predict_op(x)
            n_correct += np.sum(p == y)
        return float(n_correct) / n_total

    def f1_score(self, X, Y):
        P = []
        for x, y in zip(X, Y):
            p = self.predict_op(x)
            P.append(p)
        Y = np.concatenate(Y)
        P = np.concatenate(P)
        return f1_score(Y, P, average=None).mean()


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)

    V = len(word2idx) + 1 # 1 is for 'UNKNOWN'
    K = len(set(flatten(Ytrain)) | set(flatten(Ytest)))
    print('vocabulary size: %d' % V)

    rnn = RNN(V=V, K=K, D=10, hidden_layer_sizes=[10])
    rnn.fit(Xtrain, Ytrain, RecurrentUnit=GRU, normalize=False)

    print('train score: %.3f%%' % (100 * rnn.score(Xtrain, Ytrain)))
    print('test  score: %.3f%%' % (100 * rnn.score(Xtest, Ytest)))

    print('train f1_score: %.3f%%' % (100 * rnn.f1_score(Xtrain, Ytrain)))
    print('test  f1_score: %.3f%%' % (100 * rnn.f1_score(Xtest, Ytest)))


if __name__ == '__main__':
    main()

