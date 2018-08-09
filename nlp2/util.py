import operator
import numpy as np

from nltk.corpus import brown
from sklearn.metrics.pairwise import pairwise_distances


KEEP_WORDS = set([
    'king', 'man', 'queen', 'woman',
    'italy', 'rome', 'france', 'paris',
    'london', 'britain', 'england',
])


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def euclidean_dist(a, b):
    return np.linalg.norm(a - b)


def cosine_dist(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

# slow version
# def find_analogies(w1, w2, w3, We, word2idx):
#     v1 = We[word2idx[w1]] # king
#     v2 = We[word2idx[w2]] # man
#     v3 = We[word2idx[w3]] # woman
#     v0 = v1 - v2 + v3 # queen

#     best_words = {}
#     for dist, name in [(euclidean_dist, 'Euclidean'), (cosine_dist, 'cosine')]:
#         best_word = ''
#         min_dist = float('inf')

#         for word, idx in word2idx.items():
#             if word not in (w1, w2, w3):
#                 v4 = We[idx]
#                 d = dist(v0, v4)
#                 if d < min_dist:
#                     min_dist = d
#                     best_word = word

#         best_words[name] = best_word
#         print('closed match by %s distance: %s' % (name, best_word))
#         print('%s - %s = %s - %s' % (w1, w2, best_word, w3))

#     return best_words


# fast version
def find_analogies(w1, w2, w3, We, word2idx, idx2word=None):
    v1 = We[word2idx[w1]] # king
    v2 = We[word2idx[w2]] # man
    v3 = We[word2idx[w3]] # woman
    v0 = v1 - v2 + v3 # queen

    V, D = We.shape
    if idx2word is None:
        idx2word = {v:k for k, v in word2idx.items()}

    best_words = {}
    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, D), We, metric=dist).reshape(V)
        idxs = distances.argsort()[:4]
        best_word = None
        for idx in idxs:
            if idx2word[idx] not in (w1, w2, w3):
                best_word = idx2word[idx]
                break
        best_words[dist] = best_word

        print('closed match by %s distance: %s' % (dist, best_word))
        print('%s - %s = %s - %s' % (w1, w2, best_word, w3))

    return best_words


def nearest_neighbors(w, We, word2idx, idx2word=None, limit=5, metric='cosine'):
    if w not in word2idx:
        print('%s not in dictionary.' % w)
        return

    if idx2word is None:
        idx2word = {v:k for k, v in word2idx.items()}

    V, D = We.shape
    v = We[word2idx[w]]
    distances = pairwise_distances(v.reshape(1, D), We, metric=metric).reshape(V)
    idxs = distances.argsort()[1:limit+1]

    print('neighbors of %s:' % w)
    for idx in idxs:
        print('\t%s' % idx2word[idx])


def get_wikipedia_data(n_files, n_vocab, by_paragraph=False):
    sentences = [[]]
    word2idx = {}
    return sentences, word2idx


def get_sentences():
    # returns 57340 sentences of the Brown corpus
    # each sentence is represented as a list of individual string tokens
    return brown.sents()


def get_sentences_with_word2idx():
    sentences = get_sentences()
    indexed_sentences = []

    i = 2
    word2idx = {'START': 0, 'END': 1}
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                word2idx[token] = i
                i += 1
            indexed_sentence.append(word2idx[token])
        indexed_sentences.append(indexed_sentence)

    print('Vocab size: %d' % i)
    return indexed_sentences, word2idx


def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
    sentences = get_sentences()
    indexed_sentences = []

    i = 2
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']

    word_idx_count = {
        0: float('inf'),
        1: float('inf'),
    }

    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i += 1

            # keep track of counts for later sorting
            idx = word2idx[token]
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

            indexed_sentence.append(idx)
        indexed_sentences.append(indexed_sentence)


    # restrict vocab size

    # set all the words I want to keep to infinity
    # so that they are included when I pick the most common words
    for word in keep_words:
        if word in word2idx:
            word_idx_count[word2idx[word]] = float('inf')

    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    
    new_idx = 0
    word2idx_trim = {}
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print('%s : %s' % (word, count))

        word2idx_trim[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1

    word2idx_trim['UNKNOWN'] = new_idx
    unknown = new_idx

    assert('START' in word2idx_trim)
    assert('END' in word2idx_trim)
    # for word in keep_words:
    #     assert(word in word2idx_trim)

    # map old idx to new idx
    sentences_trim = []
    for sentence in indexed_sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_trim.append(new_sentence)

    return sentences_trim, word2idx_trim

