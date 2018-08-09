# word2vec: skip-gram + negative sampling

import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from util import get_wikipedia_data
from util import get_sentences_with_word2idx
from util import get_sentences_with_word2idx_limit_vocab
from util import find_analogies as _find_analogies


TEXT8_PATH = '../dataset/text8'
WE_FILE = 'w2v_model.npz'
W2I_FILE = 'w2v_word2idx.json'

def get_text8(file_path=TEXT8_PATH):
    words = open(file_path).read()
    word2idx = {}
    sents = [[]]

    count = 0
    for word in words.split():
        if word not in word2idx:
            word2idx[word] = count
            count += 1
        sents[0].append(word2idx[word])
    print('vocab size: %d' % count)
    return sents, word2idx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_weights(shape):
    W = np.random.randn(*shape) / np.sqrt(sum(shape))
    return W.astype(np.float32)


class Model(object):

    def __init__(self, D, V, context_sz):
        self.D = D # embedding dimension
        self.V = V # vocab size
        # NOTE: we will look context_sz to the right and context_sz to
        #       the left, so the total number of targets is 2*context_sz
        self.context_sz = context_sz

    # def _get_pnw(self, X):
    #     # calculate Pn(w) - probability distribution for negative sampling
    #     # basically just the word probability ^ 3/4
    #     word_freq = {} # count map: {index : count}
    #     word_count = sum(len(x) for x in X)

    #     for x in X:
    #         for xj in x:
    #             if xj not in word_freq:
    #                 word_freq[xj] = 0
    #             word_freq[xj] += 1

    #     self.Pnw = np.zeros(self.V)

    #     # 0 and 1 are the START and END tokens, we won't use those here
    #     for j in range(2, self.V):
    #         self.Pnw[j] = (word_freq[j] / float(word_count))**0.75

    #     assert(np.all(self.Pnw[2:] > 0))

    #     return self.Pnw

    def _get_pnw(self, X):
        # calculate Pn(w) - probability distribution for negative sampling
        # basically just the word probability ^ 3/4
        self.Pnw = np.zeros(self.V)

        for x in X:
            for xj in x:
                self.Pnw[xj] += 1

        self.Pnw = self.Pnw ** 0.75
        self.Pnw = self.Pnw / self.Pnw.sum()

        assert(np.all(self.Pnw[2:] > 0))

        return self.Pnw

    def _get_negative_samples(self, context, num_neg_samples):
        # temporarily save context values because we don't want to negative sample these
        saved = {}
        for context_idx in context:
            saved[context_idx] = self.Pnw[context_idx]
            self.Pnw[context_idx] = 0

        neg_samples = np.random.choice(
            self.V,
            size=num_neg_samples,
            replace=False,
            p=self.Pnw / np.sum(self.Pnw)
        )

        for j, pnwj in saved.items():
            self.Pnw[j] = pnwj

        assert(np.all(self.Pnw[2:] > 0))

        return neg_samples

    def fit(self, X, num_neg_samples=10, learning_rate=1e-4, mu=0.99, reg=0.1, epochs=10):
        N = len(X)
        V = self.V
        D = self.D
        self._get_pnw(X)

        # initialize weights and momentum changes
        self.W1 = init_weights((V, D))
        self.W2 = init_weights((D, V))
        dW1 = np.zeros(self.W1.shape)
        dW2 = np.zeros(self.W2.shape)

        costs = []
        cost_per_epoch = []
        sample_indices = np.arange(N)

        for i in range(epochs):
            t0 = datetime.now()
            cost_per_epoch_i = []
            np.random.shuffle(sample_indices)

            for it in range(N):
                j = sample_indices[it]
                x = X[j] # one sentence

                # too short to do 1 iteration, skip
                n = len(x)
                if n < 2 * self.context_sz + 1:
                    continue

                cost_j = []
                # for idx in range(n):
                ##### try one random window per sentence #####
                idx = np.random.choice(n)

                # do the updates manually
                Z = self.W1[x[idx], :] # NOTE: paper uses linear activation function

                start = max(0, idx - self.context_sz)
                end = min(n, idx + 1 + self.context_sz)
                context = np.concatenate([x[start:idx], x[(idx+1):end]])

                # NOTE: context can contain DUPLICATES!
                # e.g. "<UNKNOWN> <UNKNOWN> cats and dogs"
                context = np.array(list(set(context)), dtype=np.int32)

                posA = Z.dot(self.W2[:, context])
                pos_pY = sigmoid(posA)

                neg_samples = self._get_negative_samples(context, num_neg_samples)

                negA = Z.dot(self.W2[:, neg_samples])
                neg_pY = sigmoid(-negA)

                cost = -np.log(pos_pY).sum() - np.log(neg_pY).sum()
                cost_j.append(cost / (num_neg_samples + len(context)))

                # positive samples
                pos_err = pos_pY - 1
                dW2[:, context] = mu * dW2[:, context] - learning_rate * (np.outer(Z, pos_err) + reg * self.W2[:, context])

                # negative samples
                neg_err = 1 - neg_pY
                dW2[:, neg_samples] = mu * dW2[:, neg_samples] - learning_rate * (np.outer(Z, neg_err) + reg * self.W2[:, neg_samples])

                self.W2[:, context] += dW2[:, context]
                self.W2[:, neg_samples] += dW2[:, neg_samples]
                # self.W2[:, context] /= np.linalg.norm(self.W2[:, context], axis=1, keepdims=True)
                # self.W2[:, neg_samples] /= np.linalg.norm(self.W2[:, neg_samples], axis=1, keepdims=True)

                # input weights
                gradW1 = pos_err.dot(self.W2[:, context].T) + neg_err.dot(self.W2[:, neg_samples].T)
                dW1[x[idx], :] = mu * dW1[x[idx], :] - learning_rate * (gradW1 + reg * self.W1[x[idx], :])

                self.W1[x[idx], :] += dW1[x[idx], :]
                # self.W1[x[idx], :] /= np.linalg.norm(self.W1[x[idx], :])

                cost_j_mean = np.mean(cost_j)
                cost_per_epoch_i.append(cost_j_mean)
                costs.append(cost_j_mean)

                if it % 500 == 0:
                    print('epoch: %d, sentence: %d / %d, cost: %f' % (i, it, N, cost_j_mean))

            epoch_cost = np.mean(cost_per_epoch_i)
            cost_per_epoch.append(epoch_cost)
            print('>>> epoch: %d, elapsed time: %s, cost: %f' % (i, (datetime.now() - t0), epoch_cost))

        plt.plot(costs)
        plt.title('Numpy costs')
        plt.show()

        plt.plot(cost_per_epoch)
        plt.title('Numpy cost at each epoch')
        plt.show()

    def save(self, filename):
        arrays = [self.W1, self.W2]
        np.savez(filename, *arrays)


def find_analogies(w1, w2, w3, use_concat=True, we_file=WE_FILE, w2i_file=W2I_FILE):
    npz = np.load(we_file)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i_file) as f:
        word2idx = json.load(f)

    V = len(word2idx)

    if use_concat:
        We = np.hstack([W1, W2.T])
    else:
        We = (W1 + W2.T) / 2

    print('V:', V)
    print('We.shape:', We.shape)
    assert(We.shape[0] == V)

    if isinstance(w1, str):
        assert(isinstance(w2, str))
        assert(isinstance(w3, str))
        _find_analogies(w1, w2, w3, We, word2idx)
    else:
        assert(len(w1) == len(w2))
        assert(len(w1) == len(w3))
        for i, j, k in zip(w1, w2, w3):
            _find_analogies(i, j, k, We, word2idx)


def main(use_brown=True):
    if use_brown:
        sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
        # sentences, word2idx = get_sentences_with_word2idx()
        # sentences, word2idx = get_text8()
    else:
        sentences, word2idx = get_wikipedia_data(n_files=1, n_vocab=2000)

    with open(W2I_FILE, 'w') as f:
        json.dump(word2idx, f)

    D = 50
    V = len(word2idx)
    context_sz = 10
    model = Model(D, V, context_sz)

    model.fit(sentences, num_neg_samples=10, learning_rate=1e-3, mu=0.9, epochs=5, reg=0.1)

    model.save(WE_FILE)


if __name__ == '__main__':
    main(use_brown=True)

    w1 = ['king', 'france', 'france', 'paris']
    w2 = ['man', 'paris', 'paris', 'france']
    w3 = ['woman', 'london', 'rome', 'italy']
    for use_concat in (True, False):
        print('** use_concat: %s' % use_concat)
        find_analogies(w1, w2, w3, use_concat, we_file=WE_FILE, w2i_file=W2I_FILE)

