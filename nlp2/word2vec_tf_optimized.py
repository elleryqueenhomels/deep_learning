# word2vec: negative sampling (fix context, replace middle word)
# drop words randomly, decay learning rate

import os
import sys
import json
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from util import get_sentences_with_word2idx_limit_vocab as get_brown


WORD2IDX_PATH = 'word2idx_tf.json'
WORD2VEC_PATH = 'word2vec_tf.npz'


# remove punctuation in python2 and python3 version
def remove_punctuation_py2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_py3(s):
    return s.translate(str.maketrans('', '', string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_py2
else:
    remove_punctuation = remove_punctuation_py3


class Model(object):

    def __init__(self, V, D):
        self.V = V # vocab size
        self.D = D # embedding dimension

    def fit(self, sentences, context_sz=5, num_negatives=5, threshold=1e-5, 
            epochs=20, learning_rate=1e-1, mu=0.9):
        # for convenience
        V = self.V
        D = self.D

        # distribution for drawing negative samples
        p_neg = self._get_negative_sampling_distribution(sentences, V)

        # for subsampling each sentence
        p_drop = 1 - np.sqrt(threshold / p_neg)

        # params
        W1 = np.random.randn(V, D).astype(np.float32) # input-to-hidden
        W2 = np.random.randn(D, V).astype(np.float32) # hidden-to-output

        # create the model
        tf_posword = tf.placeholder(tf.int32, shape=(None, ))
        tf_negword = tf.placeholder(tf.int32, shape=(None, ))
        tf_context = tf.placeholder(tf.int32, shape=(None, ))
        tfW1 = tf.Variable(W1)
        tfW2 = tf.Variable(W2.T)

        def dot(A, B):
            C = A * B
            return tf.reduce_sum(C, axis=1)

        # correct middle word output
        emb_input  = tf.nn.embedding_lookup(tfW1, tf_posword) # (1, D)
        emb_output = tf.nn.embedding_lookup(tfW2, tf_context) # (N, D)
        correct_output = dot(emb_input, emb_output) # (N, )
        # emb_input = tf.transpose(emb_input, [1, 0])
        # correct_output = tf.matmul(emb_output, emb_input)
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones(tf.shape(correct_output)),
            logits=correct_output
        )

        # incorrect middle output
        emb_input  = tf.nn.embedding_lookup(tfW1, tf_negword) # (1, D)
        incorrect_output = dot(emb_input, emb_output) # (N, )
        # emb_input = tf.transpose(emb_input, [1, 0])
        # incorrect_output = tf.matmul(emb_output, emb_input)
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros(tf.shape(incorrect_output)),
            logits=incorrect_output
        )

        # total loss
        loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)

        # optimizer
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(loss)
        # train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # save the costs to plot them per iteration
        costs = []

        # train the model
        for epoch in range(epochs):
            np.random.shuffle(sentences)

            cost = 0
            counter = 0
            poswords = []
            negwords = []
            contexts = []
            t0 = datetime.now()

            for sentence in sentences:
                # keep only certain words based on p_drop
                sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]

                if len(sentence) < 2:
                    continue

                # randomly order words so we don't always see
                # samples in the same order
                randomly_ordered_positions = np.random.choice(
                    len(sentence),
                    size=len(sentence),
                    replace=False,
                )

                for pos in randomly_ordered_positions:
                    # the middle word
                    pos_word = sentence[pos]

                    # get the positive context words / negative samples
                    context_words = self._get_context(pos, sentence, context_sz)
                    neg_word = np.random.choice(V, p=p_neg)

                    n = len(context_words)
                    poswords += [pos_word] * n
                    negwords += [neg_word] * n
                    contexts += context_words

                    # SGD
                    # _, c = sess.run(
                    #     [train_op, loss],
                    #     feed_dict={
                    #         tf_posword: [pos_word],
                    #         tf_negword: [neg_word],
                    #         tf_context: context_words,
                    #     }
                    # )
                    # cost += c

                if len(poswords) >= 128:
                    _, c = sess.run(
                        [train_op, loss],
                        feed_dict={
                            tf_posword: poswords,
                            tf_negword: negwords,
                            tf_context: contexts,
                        }
                    )
                    cost += c

                    # reset
                    poswords = []
                    negwords = []
                    contexts = []

                counter += 1
                if counter % 100 == 0:
                    print('processed %d / %d' % (counter, len(sentences)))

            dt = datetime.now() - t0
            print('epoch: %d, cost: %f, elapsed time: %s' % (epoch, cost, dt))

            # save the cost
            costs.append(cost)

        self.W1 = sess.run(tfW1)
        self.W2 = sess.run(tfW2).T

        # plot the cost per iteration
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
        params = [self.W1, self.W2]
        np.savez(path, *params)

    def load_model(self, path):
        npz = np.load(path)
        self.W1 = npz['arr_0']
        self.W2 = npz['arr_1']
        self.V = self.W1.shape[0]
        self.D = self.W1.shape[1]
        return self.W1, self.W2


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


def get_word2idx_word2vec(w2i_path, w2v_path, get_corpus):
    exists = os.path.exists
    if exists(w2i_path) and exists(w2v_path):
        with open(w2i_path) as f:
            word2idx = json.load(f)
        model = Model(V=0, D=0) # what V and D are doesn't matter
        model.load_model(w2v_path)
        return word2idx, model.W1, model.W2

    sentences, word2idx = get_corpus()

    vocab_size = len(word2idx)
    model = Model(V=vocab_size, D=50)
    model.fit(sentences)

    with open(w2i_path, 'w') as f:
        json.dump(word2idx, f)

    model.save_model(w2v_path)

    return word2idx, model.W1, model.W2


def main():
    word2idx, W1, W2 = get_word2idx_word2vec(WORD2IDX_PATH, WORD2VEC_PATH, get_brown)
    idx2word = {i:w for w, i in word2idx.items()}
    for We in (W1, (W1 + W2.T) / 2):
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

