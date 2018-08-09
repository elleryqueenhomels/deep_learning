import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import init_weight
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import static_rnn as get_rnn_output


DATA_DIR = '../data_set/chunking'


class RNN(object):

    def __init__(self, V, D, K, hidden_layer_sizes):
        self.V = V
        self.D = D
        self.K = K
        self.sess = None
        self.hidden_layer_sizes = hidden_layer_sizes

    def set_session(self, sess):
        self.sess = sess

    def fit(self, X, Y, learning_rate=1e-2, mu=0.99, epochs=20, batch_sz=32, 
            activation=tf.nn.relu, RecurrentUnit=BasicRNNCell, show_fig=True):
        V = self.V
        D = self.D
        K = self.K
        M = self.hidden_layer_sizes[-1]
        sequence_len = X.shape[1]

        # inputs
        inputs = tf.placeholder(tf.int32, shape=(None, sequence_len))
        targets = tf.placeholder(tf.int32, shape=(None, sequence_len))
        num_samples = tf.shape(inputs)[0]
        self.inputs = inputs
        self.targets = targets

        # embedding
        We = init_weight(V, D).astype(np.float32)

        # output layer
        Wo = init_weight(M, K).astype(np.float32)
        bo = np.zeros(K, dtype=np.float32)

        # make them tensorflow variables
        tfWe = tf.Variable(We)
        tfWo = tf.Variable(Wo)
        tfbo = tf.Variable(bo)

        # make the rnn units
        rnn_units = []
        for idx, layer_size in enumerate(self.hidden_layer_sizes):
            rnn_units.append(RecurrentUnit(num_units=layer_size, 
                activation=activation, name='rnn_layer_%d' % idx))

        # get the output
        x = tf.nn.embedding_lookup(tfWe, inputs)

        # converts x from a tensor of shape N x T x D
        # into a list of length T, where each element is
        # a tensor of shape N x D
        x = tf.unstack(x, num=sequence_len, axis=1)

        # get the rnn output
        z = x
        for rnn_unit in rnn_units:
            z, states = get_rnn_output(rnn_unit, z, dtype=tf.float32)
        outputs = z


        # outputs are now of shape (T, N, M), M is the last hidden layer size
        # so make it (N, T, M)
        # and then reshape (N*T, M)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [num_samples * sequence_len, M])

        # final dense layer
        logits = tf.matmul(outputs, tfWo) + tfbo # (N*T, K)
        predictions = tf.argmax(logits, axis=1) # (N*T, )
        predict_op = tf.reshape(predictions, [num_samples, sequence_len]) # (N, T)
        labels_flat = tf.reshape(targets, [-1]) # (N*T, )
        self.predict_op = predict_op

        cost_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels_flat
            )
        )

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

        # init stuff
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        sess = self.sess
        sess.run(tf.global_variables_initializer())

        # training loop
        costs = []
        n_batches = len(X) // batch_sz
        for i in range(epochs):
            cost = 0
            n_total = 0
            n_correct = 0

            t0 = datetime.now()
            X, Y = shuffle(X, Y)

            for j in range(n_batches):
                x = X[j*batch_sz:(j+1)*batch_sz]
                y = Y[j*batch_sz:(j+1)*batch_sz]

                c, p, _ = sess.run(
                    [cost_op, predict_op, train_op],
                    feed_dict={inputs: x, targets: y}
                )
                cost += c

                # calculate the accuracy
                for yi, pi in zip(y, p):
                    # we ignore the padded entries by TF which are zero paddings
                    yii = yi[yi > 0]
                    pii = pi[pi > 0]
                    n_total += len(yii)
                    n_correct += np.sum(yii == pii)

                # print stuff periodically
                if j % 10 == 0:
                    print(
                        'epoch: %d, batch: %d/%d, correct rate so far: %.3f%%, cost so far: %.3f' % 
                        (i, j, n_batches, n_correct / n_total * 100, cost)
                    )

            costs.append(cost)
            print('>>> epoch: %d, cost: %.3f, correct rate: %.3f%%, elpased time: %s' % 
                  (i, cost, n_correct / n_total * 100, datetime.now() - t0))

        plt.plot(costs)
        plt.title('Costs')
        plt.show()

    def predict(self, X):
        return self.sess.run(self.predict_op, feed_dict={self.inputs: X})

    def score(self, X, Y):
        n_total = 0
        n_correct = 0
        P = self.predict(X)
        for y, p in zip(Y, P):
            yi = y[y > 0]
            pi = p[p > 0]
            n_total += len(yi)
            n_correct += np.sum(yi == pi)
        return float(n_correct) / n_total


def flatten(lists):
    return [item for sublist in lists for item in sublist]


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
    word_idx = 1
    tag_idx = 1
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
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)

    V = len(word2idx) + 2 # vocab size (+1 for unknown, +1 b/c start from 1)
    K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1 # num classes/tags (+1 b/c start from 1)
    sequence_len = max(len(x) for x in Xtrain + Xtest)

    # pad sequences
    Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=sequence_len)
    Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=sequence_len)
    Xtest = tf.keras.preprocessing.sequence.pad_sequences(Xtest, maxlen=sequence_len)
    Ytest = tf.keras.preprocessing.sequence.pad_sequences(Ytest, maxlen=sequence_len)
    print('Xtrain.shape = %s' % str(Xtrain.shape))
    print('Xtest.shape = %s' % str(Ytest.shape))

    # create model
    rnn = RNN(V=V, K=K, D=10, hidden_layer_sizes=[10])
    rnn.fit(Xtrain, Ytrain, learning_rate=1e-2, mu=0.99, epochs=20, batch_sz=32, 
            activation=tf.nn.relu, RecurrentUnit=GRUCell, show_fig=True)

    print('Train score: %.3f%%' % (100 * rnn.score(Xtrain, Ytrain)))
    print('Test score: %.3f%%' % (100 * rnn.score(Xtest, Ytest)))


if __name__ == '__main__':
    main()

