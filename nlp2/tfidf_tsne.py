import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer

from datetime import datetime
from util import find_analogies
from util import get_wikipedia_data
from util import get_sentences_with_word2idx
from util import get_sentences_with_word2idx_limit_vocab


def main():
    analogies_to_try = (
        ('king', 'man', 'woman'),
        ('france', 'paris', 'london'),
        ('france', 'paris', 'rome'),
        ('paris', 'france', 'italy'),
    )

    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=1500)
    # sentences, word2idx = get_wikipedia_data(n_files=20, n_vocab=2000, by_paragraph=True)
    # with open('tfidf_word2idx.json', 'w') as f:
    #     json.dump(word2idx, f)

    notfound = False
    for word_list in analogies_to_try:
        for w in word_list:
            if w not in word2idx:
                print('%s not in vocab, remove it from analogies_to_try or increase vocab size' % w)
                notfound = True
    if notfound:
        exit()

    # build term document matrix
    V = len(word2idx)
    N = len(sentences)

    print('Vocab size: %d' % V)
    print('Documents num: %d' % N)

    # create raw counts first
    A = np.zeros((V, N))
    for d, sentence in enumerate(sentences):
        for t in sentence:
            A[t, d] += 1
    print('>>> Finished getting raw counts')

    transformer = TfidfTransformer()
    A = transformer.fit_transform(A.T).T
    A = A.toarray()

    idx2word = {v:k for k, v in word2idx.items()}

    # plot the data in 2-D
    tsne = TSNE()
    Z = tsne.fit_transform(A)

    plt.scatter(Z[:,0], Z[:,1])
    for i in range(V):
        plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i,1]))
    plt.draw()

    # create a higher-D word embedding, try word analogies
    # tsne = TSNE(n_components=3)
    # We = tsne.fit_transform(A)
    We = Z

    for word_list in analogies_to_try:
        w1, w2, w3 = word_list
        find_analogies(w1, w2, w3, We, word2idx)

    plt.show() # pause script until plot is closed


if __name__ == '__main__':
    main()

