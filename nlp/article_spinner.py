# Very basic article spinner using trigram

import nltk
import random
from bs4 import BeautifulSoup


# load the reviews, data from:
# http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('../data_set/electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')


# extract trigrams and insert into dictionary
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i + 2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i + 1])

# turn each array of middle-words into a probability vector
trigram_probabilities = {}
for k, words in trigrams.items():
    # create a dictionary of word -> count
    if len(set(words)) > 1:
        dic = {}
        cnt = 0
        for w in words:
            if w not in dic:
                dic[w] = 0
            dic[w] += 1
            cnt += 1
        for w, c in dic.items():
            dic[w] = float(c) / cnt
        trigram_probabilities[k] = dic


def random_sample(dic):
    # choose a random sample from dictionary {word : prob}
    t = random.random()
    cumulative = 0
    for w, p in dic.items():
        cumulative += p
        if cumulative > t:
            return w


def tokens_to_review(tokens):
    review = ' '.join(tokens)
    review = review.replace(' .', '.')
    review = review.replace(' ,', ',')
    review = review.replace(' !', '!')
    review = review.replace(' ?', '?')
    review = review.replace(' %', '%')
    review = review.replace('$ ', '$')
    review = review.replace(' \'', '\'')
    return review


def article_spinner(review, replace_prob=0.3):
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < replace_prob:
            k = (tokens[i], tokens[i + 2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i + 1] = w

    print('Origin review:\n%s\n' % s)
    print('Spun review:\n%s\n' % tokens_to_review(tokens))


if __name__ == '__main__':
    review = random.choice(positive_reviews)
    article_spinner(review, replace_prob=1.0)

