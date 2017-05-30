# Utility

import numpy as np
import theano
import theano.tensor as T
import operator
import string
import os

from nltk import pos_tag, word_tokenize


def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def get_activation(activation_type=1):
	if activation_type == 1:
		return T.nnet.relu
	elif activation_type == 2:
		return T.tanh
	else:
		return T.nnet.sigmoid


def all_parity_pairs(nbit):
	# total number of samples (Ntotal) will be a multiple of 100
	N = 2**nbit
	remainder = 100 - (N % 100)
	Ntotal = N + remainder
	X = np.zeros((Ntotal, nbit))
	Y = np.zeros(Ntotal)
	for ii in range(Ntotal):
		i = ii % N
		# now generate the i-th sample
		for j in range(nbit):
			if i % (2**(j+1)) != 0:
				i -= 2**j
				X[ii,j] = 1
		Y[ii] = X[ii].sum() % 2
	return X, Y


def all_parity_pairs_with_sequence_labels(nbit):
	X, Y = all_parity_pairs(nbit) # X: (N, D), Y:(N, )
	N, T = X.shape # T == D, consider each row of X as a sequence of length T

	# we want every time step to have a label
	Y_T = np.zeros(X.shape, dtype=np.int32)
	for n in range(N):
		ones_count = 0
		for t in range(T):
			if X[n,t] == 1:
				ones_count += 1
			if ones_count % 2 == 1:
				Y_T[n,t] = 1

	X = X.reshape(N, T, 1).astype(np.float32)
	return X, Y_T


def remove_punctuation(s):
	# return s.translate(None, string.punctuation) # python 2.7
	translator = str.maketrans('', '', string.punctuation) # python 3.6
	return s.translate(translator)


def get_robert_frost():
	word2idx = {'START': 0, 'END': 1}
	current_idx = 2
	sentences = []
	for line in open('../data_set/robert_frost.txt'):
		line = line.strip()
		if line:
			tokens = remove_punctuation(line.lower()).split()
			sentence = []
			for t in tokens:
				if t not in word2idx:
					word2idx[t] = current_idx
					current_idx += 1
				idx = word2idx[t]
				sentence.append(idx)
			sentences.append(sentence)
	return sentences, word2idx


def get_tags(s):
	tuples = pos_tag(word_tokenize(s))
	return [y for x, y in tuples]


def get_poetry_classifier_data(samples_per_class=None, load_cached=True, save_cached=True):
	datafile = 'poetry_classifier_data.npz'
	if load_cached and os.path.exists(datafile):
		npz = np.load(datafile)
		X = npz['arr_0']
		Y = npz['arr_1']
		V = int(npz['arr_2'])
		return X, Y, V

	limit = (samples_per_class is not None)

	word2idx = {}
	current_idx = 0
	X = []
	Y = []
	for fn, label in zip(('../data_set/robert_frost.txt', '../data_set/edgar_allan_poe.txt'), (0, 1)):
		count = 0
		for line in open(fn):
			line = line.strip()
			if line:
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
					# quit early because the tokenizer is very slow
					if limit and count >= samples_per_class:
						break

	if save_cached:
		np.savez(datafile, X, Y, current_idx)

	return X, Y, current_idx


def my_tokenizer(s):
	s = remove_punctuation(s)
	s = s.lower() # downcase
	return s.split()


def get_wikipedia_data(n_files, n_vocab, path='../data_set/', by_paragraph=False):
	input_files = [f for f in os.listdir(path) if f.startswith('enwiki') and f.endswith('txt')]

	# return variables
	sentences = []
	word2idx = {'START': 0, 'END': 1}
	idx2word = ['START', 'END']
	current_idx = 2
	word_idx_count = {0: float('inf'), 1: float('inf')}

	if n_files is not None:
		input_files = input_files[:n_files]

	for f in input_files:
		print('reading: %s' % f)
		for line in open(path + f):
			line = line.strip()
			# do not count headers, structured data, lists, etc...
			if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
				if by_paragraph:
					sentence_lines = [line]
				else:
					sentence_lines = line.split('. ')
				for sentence in sentence_lines:
					tokens = my_tokenizer(sentence)
					for t in tokens:
						if t not in word2idx:
							word2idx[t] = current_idx
							idx2word.append(t)
							current_idx += 1
						idx = word2idx[t]
						word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
					sentence_by_idx = [word2idx[t] for t in tokens]
					sentences.append(sentence_by_idx)

	# restrict vocab size
	sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
	word2idx_small = {}
	new_idx = 0
	idx_new_idx_map = {}
	for idx, count in sorted_word_idx_count[:n_vocab]:
		word = idx2word[idx]
		print('%s, %s' % (word, count))
		word2idx_small[word] = new_idx
		idx_new_idx_map[idx] = new_idx
		new_idx += 1
	# let 'unknown' be the last token
	word2idx_small['UNKNOWN'] = new_idx
	unknown = new_idx

	# sanity check
	assert('START' in word2idx_small)
	assert('END' in word2idx_small)
	assert('king' in word2idx_small)
	assert('queen' in word2idx_small)
	assert('man' in word2idx_small)
	assert('woman' in word2idx_small)

	# map old idx to new idx
	sentences_small = []
	for sentence in sentences:
		if len(sentence) > 1:
			new_sentence = [idx_new_idx_map.get(idx, unknown) for idx in sentence]
			sentences_small.append(new_sentence)

	return sentences_small, word2idx_small

