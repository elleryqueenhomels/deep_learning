import nltk
import numpy as np

from bayes_classifier import Bayes
from naive_bayes import NaiveBayes
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.strip() for w in open('../data_set/stopwords.txt'))

# load the reviews, data from:
# http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('../data_set/electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('../data_set/electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# balance the dataset
min_len = min(len(positive_reviews), len(negative_reviews))

np.random.shuffle(positive_reviews)
np.random.shuffle(negative_reviews)
positive_reviews = positive_reviews[:min_len]
negative_reviews = negative_reviews[:min_len]

# custom our own tokenizer
def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


# create a word-to-index map so that we can create our word-frequency vector
# let's also save the tokenized versions so we don't need to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

def collect_tokens(reviews, tokenized_list):
    global current_index
    for review in reviews:
        tokens = my_tokenizer(review.text)
        tokenized_list.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1

collect_tokens(positive_reviews, positive_tokenized)
collect_tokens(negative_reviews, negative_tokenized)


# now let's create our input metrices
def tokens_to_vector(tokens, label):
    v = np.zeros(len(word_index_map) + 1) # last elem is for label
    for token in tokens:
        idx = word_index_map[token]
        v[idx] += 1
    v = v / v.sum() # normalize
    v[-1] = label
    return v

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))

idx = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[idx] = xy
    idx += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[idx] = xy
    idx += 1

# shuffle the data and create train/test splits
np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]


model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print('\nClassification rate for LogisticRegression: %f\n' % model.score(Xtest, Ytest))

# let's look at the weights for each word
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if abs(weight) > threshold:
        print('word: %s, weight: %f' % (word, weight))


# model = Bayes()
model = NaiveBayes()
model.fit(Xtrain, Ytrain)
print('\nClassification rate for Bayes: %f\n' % model.score(Xtest, Ytest))

