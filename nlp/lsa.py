# Latent Semantic Analysis visualization

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


titles = [line.rstrip() for line in open('../data_set/all_book_titles.txt')]

stopwords = set(w.rstrip() for w in open('../data_set/stopwords.txt'))

# NOTE: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })

wordnet_lemmatizer = WordNetLemmatizer()

tag_map = {'J': wordnet.ADJ, 
           'V': wordnet.VERB, 
           'N': wordnet.NOUN, 
           'R': wordnet.ADV}

def my_lemmatizer(tokens):
    result = []
    word_tags = nltk.pos_tag(tokens)
    for word, tag in word_tags:
        if tag[0] in tag_map:
            t = tag_map[tag[0]]
            word = wordnet_lemmatizer.lemmatize(word, t)
        result.append(word)
    return result

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = my_lemmatizer(tokens)
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors
# let's also save the tokenized versions so we don't have to tokenize again
word_index_map = {}
index_word_map = []
current_index = 0
all_tokens = []

for title in titles:
    tokens = my_tokenizer(title)
    all_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            index_word_map.append(token)
            current_index += 1


# create our input matrices - just indicator variables for 
# this example - works better than proportions
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        idx = word_index_map[t]
        x[idx] = 1
    return x

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N)) # term-document matrix, not document-term matrix

for i, tokens in enumerate(all_tokens):
    X[:,i] = tokens_to_vector(tokens)


def main():
    model = PCA()
    # model = TSNE()
    # model = TruncatedSVD()
    Z = model.fit_transform(X)

    print('X.shape =', X.shape)
    print('Z.shape =', Z.shape)

    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
    plt.show()


if __name__ == '__main__':
    main()

