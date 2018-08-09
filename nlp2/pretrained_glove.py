# GET THE PRETRAINED VECTORS:
# GloVe: https://nlp.stanford.edu/projects/glove/
# Direct link: http://nlp.stanford.edu/data/glove.6B.zip

import numpy as np
from util import find_analogies, nearest_neighbors


GLOVE_PATH = '../data_set/glove.6B/glove.6B.50d.txt'
# METRIC = 'euclidean'
METRIC = 'cosine'


def load_glove(file_path=GLOVE_PATH):
    print('Loading word vectors...')

    embedding = []
    word2idx = {}
    idx2word = []

    with open(file_path, encoding='utf-8') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype=np.float32)
            embedding.append(vec)
            idx2word.append(word)

    embedding = np.array(embedding)
    word2idx = {w:i for i, w in enumerate(idx2word)}

    print('Found %d word vectors.' % len(idx2word))

    return embedding, word2idx, idx2word


def main():
    embedding, word2idx, idx2word = load_glove()

    find_analogies('king', 'man', 'woman', embedding, word2idx, idx2word)
    find_analogies('france', 'paris', 'london', embedding, word2idx, idx2word)
    find_analogies('france', 'paris', 'rome', embedding, word2idx, idx2word)
    find_analogies('paris', 'france', 'italy', embedding, word2idx, idx2word)
    find_analogies('france', 'french', 'english', embedding, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'chinese', embedding, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'italian', embedding, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'australian', embedding, word2idx, idx2word)
    find_analogies('december', 'november', 'june', embedding, word2idx, idx2word)
    find_analogies('miami', 'florida', 'texas', embedding, word2idx, idx2word)
    find_analogies('einstein', 'scientist', 'painter', embedding, word2idx, idx2word)
    find_analogies('china', 'rice', 'bread', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'she', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'aunt', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'sister', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'wife', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'actress', embedding, word2idx, idx2word)
    find_analogies('man', 'woman', 'mother', embedding, word2idx, idx2word)
    find_analogies('heir', 'heiress', 'princess', embedding, word2idx, idx2word)
    find_analogies('nephew', 'niece', 'aunt', embedding, word2idx, idx2word)
    find_analogies('france', 'paris', 'tokyo', embedding, word2idx, idx2word)
    find_analogies('france', 'paris', 'beijing', embedding, word2idx, idx2word)
    find_analogies('february', 'january', 'november', embedding, word2idx, idx2word)
    find_analogies('france', 'paris', 'rome', embedding, word2idx, idx2word)
    find_analogies('paris', 'france', 'italy', embedding, word2idx, idx2word)

    nearest_neighbors('king', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('france', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('japan', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('einstein', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('woman', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('nephew', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('february', embedding, word2idx, idx2word, metric=METRIC)
    nearest_neighbors('rome', embedding, word2idx, idx2word, metric=METRIC)


if __name__ == '__main__':
    main()

