# GET THE PRETRAINED VECTORS:
# word2vec: https://code.google.com/archive/p/word2vec/
# Direct link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

import numpy as np
from gensim.models import KeyedVectors


W2V_PATH = '../data_set/GoogleNews-vectors-negative300.bin'


def load_word2vec(file_path=W2V_PATH):
    print('Loading word vectors...')

    # 3 million words and phrases, dim = 300
    word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=True)

    return word_vectors


def find_analogies(w1, w2, w3, word_vectors):
    r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    print('%s - %s = %s - %s' % (w1, w2, r[0][0], w3))


def nearest_neighbors(w, word_vectors):
    r = word_vectors.most_similar(positive=[w])
    print('neighbors of %s:' % w)
    for word, score in r:
        print('\t%s' % word)


def main():
    word_vectors = load_word2vec()

    find_analogies('king', 'man', 'woman', word_vectors)
    find_analogies('france', 'paris', 'london', word_vectors)
    find_analogies('france', 'paris', 'rome', word_vectors)
    find_analogies('paris', 'france', 'italy', word_vectors)
    find_analogies('france', 'french', 'english', word_vectors)
    find_analogies('japan', 'japanese', 'chinese', word_vectors)
    find_analogies('japan', 'japanese', 'italian', word_vectors)
    find_analogies('japan', 'japanese', 'australian', word_vectors)
    find_analogies('december', 'november', 'june', word_vectors)
    find_analogies('miami', 'florida', 'texas', word_vectors)
    find_analogies('einstein', 'scientist', 'painter', word_vectors)
    find_analogies('china', 'rice', 'bread', word_vectors)
    find_analogies('man', 'woman', 'she', word_vectors)
    find_analogies('man', 'woman', 'aunt', word_vectors)
    find_analogies('man', 'woman', 'sister', word_vectors)
    find_analogies('man', 'woman', 'wife', word_vectors)
    find_analogies('man', 'woman', 'actress', word_vectors)
    find_analogies('man', 'woman', 'mother', word_vectors)
    find_analogies('heir', 'heiress', 'princess', word_vectors)
    find_analogies('nephew', 'niece', 'aunt', word_vectors)
    find_analogies('france', 'paris', 'tokyo', word_vectors)
    find_analogies('france', 'paris', 'beijing', word_vectors)
    find_analogies('february', 'january', 'november', word_vectors)
    find_analogies('france', 'paris', 'rome', word_vectors)
    find_analogies('paris', 'france', 'italy', word_vectors)

    nearest_neighbors('king', word_vectors)
    nearest_neighbors('france', word_vectors)
    nearest_neighbors('japan', word_vectors)
    nearest_neighbors('einstein', word_vectors)
    nearest_neighbors('woman', word_vectors)
    nearest_neighbors('nephew', word_vectors)
    nearest_neighbors('february', word_vectors)
    nearest_neighbors('rome', word_vectors)


if __name__ == '__main__':
    main()

