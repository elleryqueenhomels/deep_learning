# Naive Bayes spam detection

from __future__ import division, print_function
from builtins import range

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


# NOTE: technically multinomial NB is for 'counts', but the documentation says
#       it will work for other types of 'counts', like tf-idf, so it should
#       also work for our 'word proportions'

data = pd.read_csv('../data_set/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]


model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Classification rate for NB:', model.score(Xtest, Ytest))


# we could use any other model
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print('Classification rate for AdaBoost:', model.score(Xtest, Ytest))


# test my own model
from naive_bayes import NaiveBayes

model = NaiveBayes()
model.fit(Xtrain, Ytrain)
print('Classification rate for my own NB:', model.score(Xtest, Ytest))


from bayes_classifier import Bayes

model = Bayes()
model.fit(Xtrain, Ytrain)
print('Classification rate for my own Bayes:', model.score(Xtest, Ytest))

