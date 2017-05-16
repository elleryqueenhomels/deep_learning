# Demonstrate how HMMs can be used for classification.

import string
import numpy as np
import matplotlib.pyplot as plt

from hmmd_theano import HMM
from sklearn.utils import shuffle
from nltk import pos_tag, word_tokenize


class HMMClassifier(object):
	def __init__(self):
		pass

	def fit(self, X, Y, V, M=5, learning_rate=1e-5, max_iter=10, p_cost=1.0, debug=False):
		K = len(set(Y)) # number of classes - assume 0..K-1
		self.models = []
		self.priors = []
		for k in range(K):
			# gather all the training data for this class
			thisX = [x for x, y in zip(X, Y) if y == k]
			C = len(thisX)
			self.priors.append(np.log(C)) # log(prior) = log(C/N) = log(C) - log(N), because log(N) is constant, so throw it out.

			hmm = HMM(M) # M is number of hidden states
			hmm.fit(thisX, V=V, learning_rate=learning_rate, max_iter=max_iter, p_cost=p_cost, debug=debug)
			self.models.append(hmm)

	def score(self, X, Y):
		N = len(Y)
		correct = 0
		for x, y in zip(X, Y):
			lls = [hmm.log_likelihood(x) + prior for hmm, prior in zip(self.models, self.priors)]
			p = np.argmax(lls)
			if p == y:
				correct += 1
		return float(correct) / N


def remove_punctuation(s):
	# return s.translate(None, string.punctuation) # python 2.7
	# return s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # not good enough
	translator = str.maketrans('', '', string.punctuation) # python 3.6
	return s.translate(translator)


def get_tags(s):
	tuples = pos_tag(word_tokenize(s))
	return [y for x, y in tuples]


def get_data(limit=100):
	word2idx = {}
	current_idx = 0
	X, Y = [], []
	for fn, label in zip(('robert_frost.txt', 'edgar_allan_poe.txt'), (0, 1)):
		count = 0
		for line in open(fn):
			line = line.rstrip()
			if line:
				# print(line)
				# tokens = remove_punctuation(line.lower()).split()
				tokens = get_tags(line)
				if len(tokens) > 1:
					# scan doesn't work nice here, technically could fix...
					for token in tokens:
						if token not in word2idx:
							word2idx[token] = current_idx
							current_idx += 1
					sequence = np.array([word2idx[w] for w in tokens])
					X.append(sequence)
					Y.append(label)
					count += 1
					# print(count)
					if count >= limit:
						break
	print('Vocabulary:', word2idx.keys())
	return X, Y, current_idx


def main():
	X, Y, V = get_data()
	print('\nFinished loading data...')
	print('len(X): %d' % len(X))
	print('Vocabulary size: %d' % V)
	X, Y = shuffle(X, Y)
	Ntrain = int(0.8 * len(X))
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	model = HMMClassifier()
	model.fit(Xtrain, Ytrain, V, M=5, learning_rate=1e-5, max_iter=100, p_cost=0.1, debug=True)
	score = model.score(Xtest, Ytest)
	print('\nScore: %.1f%%\n' % (score*100))


if __name__ == '__main__':
	main()

