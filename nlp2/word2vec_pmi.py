# word2vec: PMI (Pointwise Mutual Information)

import os
import sys
import json
import string
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.sparse import load_npz
from scipy.sparse import save_npz
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from util import get_sentences_with_word2idx_limit_vocab as get_brown

LATENT_DIM = 100
WORD2IDX_PATH = 'word2idx_pmi.json'
WORD2VEC_PATH = 'word2vec_pmi.npz'
PMI_MATRIX = 'pmi_matrix.npz'


class Model(object):

    def __init__(self, V, D):
        self.V = V # vocab size
        self.D = D # embedding dimension

    def fit(self, sentences, pmi_matrix=None, context_sz=10, reg=1e-1, epochs=20):
        # for convenience
        V = self.V
        D = self.D
        N = len(sentences)

        if not os.path.exists(pmi_matrix):
            # init counts
            wc_counts = lil_matrix((V, V))

            ### make PMI matrix
            for it, sentence in enumerate(sentences):
                if it % 10000 == 0:
                    print('processed: %d / %d' % (it, N))

                n = len(sentence)
                for i in range(n):
                    # i is not the word index
                    # j is not the word index
                    # i just points to the i-th element of the sequence(sentence) we are looking at
                    wi = sentence[i]

                    start = max(0, i - context_sz)
                    end = min(n, i + 1 + context_sz)

                    for j in range(start, i):
                        wj = sentence[j]
                        wc_counts[wi, wj] += 1
                    for j in range(i+1, end):
                        wj = sentence[j]
                        wc_counts[wi, wj] += 1

            print('>>>>> Finished counting')

            save_npz(pmi_matrix, csr_matrix(wc_counts))
        else:
            wc_counts = load_npz(pmi_matrix)

        # context counts get raised ^ 0.75
        c_counts = wc_counts.sum(axis=0).A.flatten()
        c_counts = (c_counts + 1) ** 0.75 # add one to avoid divide by zero
        c_probs = c_counts / c_counts.sum()
        c_probs = c_probs.reshape(1, V)

        assert(np.all(c_probs > 0))

        # PMI(x, y) = P(x, y) / ( P(x) * P(y) )
        # P(w, c) = #(w, c) / #(total)
        # P(w) = #(w) / #(total)
        # PMI(w, c) = P(w, c) / ( P(w) * P(c)) = #(w, c) / (#(w) * P(c))
        w_counts = wc_counts.sum(axis=1) + 1 # add one to avoid divide by zero
        pmi = wc_counts / w_counts / c_probs
        logX = np.log(pmi.A + 1)
        logX[logX < 0] = 0

        assert(np.all(logX >= 0))

        print('type(pmi): %s' % type(pmi))
        print('type(logX): %s' % type(logX))

        ### do Alternating Least Squares

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()


        costs = []
        t0 = datetime.now()

        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = (delta * delta).sum()
            costs.append(cost)

            ### partially vectorized updates ###
            # # update W
            # matrix = reg * np.eye(D) + U.T.dot(U)
            # for i in range(V):
            #     vector = (logX[i,:] - b[i] - c - mu).dot(U)
            #     W[i] = np.linalg.solve(matrix, vector)

            # # update b
            # for i in range(V):
            #     numerator = (logX[i,:] - W[i].dot(U.T) - c - mu).sum()
            #     b[i] = numerator / V #/ (1 + reg)

            # # update U
            # matrix = reg * np.eye(D) + W.T.dot(W)
            # for j in range(V):
            #     vector = (logX[:,j] - b - c[j] - mu).dot(W)
            #     U[j] = np.linalg.solve(matrix, vector)

            # # update c
            # for j in range(V):
            #     numerator = (logX[:,j] - W.dot(U[j]) - b - mu).sum()
            #     c[j] = numerator / V #/ (1 + reg)

            ### vectorized updates ###
            # vectorized update W
            matrix = reg * np.eye(D) + U.T.dot(U)
            vector = (logX - b.reshape(V, 1) - c.reshape(1, V) - mu).dot(U).T
            W = np.linalg.solve(matrix, vector).T

            # vectorized update b
            b = (logX - W.dot(U.T) - c.reshape(1, V) - mu).sum(axis=1) / V

            # vectorized update U
            matrix = reg * np.eye(D) + W.T.dot(W)
            vector = (logX - b.reshape(V, 1) - c.reshape(1, V) - mu).dot(W).T
            U = np.linalg.solve(matrix, vector).T

            # vectorized update c
            c = (logX - W.dot(U.T) - b.reshape(V, 1) - mu).sum(axis=0) / V

        print('>>> train duration: %s' % (datetime.now() - t0))

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()


    def _get_negative_sampling_distribution(self, sentences, vocab_size):
        # Pn(w) = prob of word occuring
        # we would like to sample the negative samples
        # such that words that occur more often
        # should be sampled more often

        word_freq = np.ones(vocab_size)
        # word_count = sum(len(sentence) for sentence in sentences)
        for sentence in sentences:
            for word in sentence:
                word_freq[word] += 1

        # smooth it
        p_neg = word_freq ** 0.75

        # normalize it
        p_neg = p_neg / p_neg.sum()

        assert(np.all(p_neg > 0))

        return p_neg

    def _get_context(self, pos, sentence, context_sz):
        start = max(0, pos - context_sz)
        end = min(len(sentence), pos + 1 + context_sz)

        context = []
        for idx in sentence[start:pos]:
            context.append(idx)
        for idx in sentence[pos+1:end]:
            context.append(idx)

        return context

    def save_model(self, path):
        params = [self.W, self.U]
        np.savez(path, *params)

    def load_model(self, path):
        npz = np.load(path)
        self.W = npz['arr_0']
        self.U = npz['arr_1']
        self.V = self.W.shape[0]
        self.D = self.W.shape[1]
        return self.W, self.U


def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, We):
    V, D = We.shape

    # don't actually use pos2 in calculation, just print what's expected
    print('\nTesting: %s - %s = %s - %s' % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print('Ooops! %s not in word2idx' % w)
            return

    p1 = We[word2idx[pos1]]
    n1 = We[word2idx[neg1]]
    p2 = We[word2idx[pos2]]
    n2 = We[word2idx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), We, metric='cosine').reshape(V)
    idxs = distances.argsort()[:10]

    # pick one that's not p1, n1 and n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for idx in idxs:
        if idx not in keep_out:
            best_idx = idx
            break

    print('>>> Got: %s - %s = %s - %s' % (pos1, neg1, idx2word[best_idx], neg2))
    print('Closest 10:')
    for idx in idxs:
        print('\t%s : %f' % (idx2word[idx], distances[idx]))

    print('dist to %s: %f' % (pos2, cos_dist(p2, vec)))


def get_word2idx_word2vec(w2i_path, w2v_path, pmi_matrix, get_corpus):
    exists = os.path.exists
    if exists(w2i_path) and exists(w2v_path):
        with open(w2i_path) as f:
            word2idx = json.load(f)
        model = Model(V=0, D=0) # what V and D are doesn't matter
        model.load_model(w2v_path)
        return word2idx, model.W, model.U

    sentences, word2idx = get_corpus()

    vocab_size = len(word2idx)
    model = Model(V=vocab_size, D=LATENT_DIM)
    model.fit(sentences, pmi_matrix)

    with open(w2i_path, 'w') as f:
        json.dump(word2idx, f)

    model.save_model(w2v_path)

    return word2idx, model.W, model.U


def main():
    word2idx, W1, W2 = get_word2idx_word2vec(WORD2IDX_PATH, WORD2VEC_PATH, PMI_MATRIX, get_brown)
    idx2word = {i:w for w, i in word2idx.items()}
    for We in (W1, (W1 + W2) / 2):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)


if __name__ == '__main__':
    main()

