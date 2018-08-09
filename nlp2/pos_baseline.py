import os
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

# suppress warnings
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = '../data_set/chunking'


class LogisticRegression:

    def __init__(self):
        pass

    def fit(self, X, Y, V=None, K=None, lr=1e-1, mu=0.99, batch_sz=100, epochs=6):
        if V is None:
            V = len(set(X))
        if K is None:
            K = len(set(Y))

        lr = np.float32(lr)
        mu = np.float32(mu)

        N = len(X)
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)

        W = np.random.randn(V, K) / np.sqrt(V + K)
        b = np.zeros(K)

        self.W = theano.shared(W.astype(np.float32))
        self.b = theano.shared(b.astype(np.float32))
        self.params = [self.W, self.b]

        thX = T.ivector('X')
        thY = T.ivector('Y')

        py_x = T.nnet.softmax(self.W[thX] + self.b)
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in self.params]

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )

        updates = [
            (p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True,
        )

        costs = []
        n_batches = N // batch_sz
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)

                if j % 200 == 0:
                    print('epoch: %d, batch: %d / %d, cost: %.3f, error_rate: %.3f%%' % 
                        (i, j, n_batches, c, np.mean(p != Ybatch) * 100))

        plt.plot(costs)
        plt.title('Costs')
        plt.show()

    def predict(self, X):
        X = np.asarray(X, dtype=np.int32)
        return self.predict_op(X)

    def score(self, X, Y):
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        P = self.predict_op(X)
        return np.mean(P == Y)

    def f1_score(self, X, Y):
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        P = self.predict_op(X)
        return f1_score(Y, P, average=None).mean()


def get_data(split_sequences=False):
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
    TEST_FILE = os.path.join(DATA_DIR, 'test.txt')

    if not os.path.exists(DATA_DIR):
        print('Data set <%s> doesnot exist!' % DATA_DIR)
        exit()
    elif not os.path.exists(TRAIN_FILE):
        print('Data set <%s> doesnot exist!' % TRAIN_FILE)
        exit()
    elif not os.path.exists(TEST_FILE):
        print('Data set <%s> doesnot exist!' % TEST_FILE)
        exit()

    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []

    for line in open(TRAIN_FILE, 'r'):
        line = line.strip()
        if line:
            word, tag, _ = line.split()

            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    print('tag size: %d' % len(tag2idx))

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []

    for line in open(TEST_FILE, 'r'):
        line = line.strip()
        if line:
            word, tag, _ = line.split()
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx) # use this as UNKNOWN
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data()

    # convert to numpy arrays
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    # convert Xtrain to indicator matrix
    N = len(Xtrain)
    V = len(word2idx) + 1
    print('vocabulary size: %d' % V)
    # Xtrain_indicator = np.zeros((N, V))
    # Xtrain_indicator[np.arange(N), Xtrain] = 1

    # train and score
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V)
    print('\n>>> lr training complete...\n')
    print('lr train score: %.3f%%' % (model.score(Xtrain, Ytrain) * 100))
    print('lr train f1_score: %.3f' % (model.f1_score(Xtrain, Ytrain)))

    # decision tree
    dt = DecisionTreeClassifier()

    # without indicator
    dt.fit(Xtrain.reshape(N, 1), Ytrain)
    p = dt.predict(Xtrain.reshape(N, 1))
    print('dt train score: %.3f%%' % (dt.score(Xtrain.reshape(N, 1), Ytrain) * 100))
    print('dt train f1_score: %.3f' % f1_score(Ytrain, p, average=None).mean())

    # with indicator -- too slow!!
    # dt.fit(Xtrain_indicator, Ytrain)
    # print('dt train score: %.3f%%' % (dt.score(Xtrain_indicator, Ytrain) * 100))


    Ntest = len(Xtest)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    # logistic test score
    print('lr test socre: %.3f%%' % (model.score(Xtest, Ytest) * 100))
    print('lr test f1_socre: %.3f' % model.f1_score(Xtest, Ytest))

    # decision tree test score
    p = dt.predict(Xtest.reshape(Ntest, 1))
    print('dt test score: %.3f%%' % (dt.score(Xtest.reshape(Ntest, 1), Ytest) * 100))
    print('dt test f1_score: %.3f' % f1_score(Ytest, p, average=None).mean())


if __name__ == '__main__':
    main()

