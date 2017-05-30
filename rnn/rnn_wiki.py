# Using RNN to create a Language Model for Wikipedia data

import sys
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from sklearn.utils import shuffle
from gru import GRU
from lstm import LSTM
from util import init_weight, get_wikipedia_data
from brown_corpus import get_sentences_with_word2idx_limit_vocab


class RNN(object):
	def __init__(self, D, hidden_layer_sizes, V, RecurrentUnit=GRU, activation=T.nnet.relu):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.D = D
		self.V = V
		self.RecurrentUnit = RecurrentUnit
		self.activation = activation

	def fit(self, X, epochs=10, learning_rate=1e-5, momentum=0.99, normalize=True, debug=False, print_period=200):
		# for convenience
		lr = learning_rate
		mu = momentum
		RecurrentUnit = self.RecurrentUnit
		activation = self.activation
		D = self.D
		V = self.V
		N = len(X)

		# variables
		We = init_weight(V, D)
		self.hidden_layers = []
		Mi = D
		for Mo in self.hidden_layer_sizes:
			ru = RecurrentUnit(Mi, Mo, activation)
			self.hidden_layers.append(ru)
			Mi = Mo

		Wo = init_weight(Mi, V)
		bo = np.zeros(V)

		self.We = theano.shared(We)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.Wo, self.bo]
		for ru in reversed(self.hidden_layers):
			self.params += ru.params

		thX = T.ivector('X')
		thY = T.ivector('Y')

		Z = self.We[thX] # shape: (T, D), T == thX.shape[0]
		for ru in self.hidden_layers:
			Z = ru.output(Z)
		py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

		prediction = T.argmax(py_x, axis=1)
		# let's return py_x too so we can draw a sample instead
		self.predict_op = theano.function(
			inputs=[thX],
			outputs=[py_x, prediction],
			allow_input_downcast=True,
		)

		cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(p.get_value() * 0) for p in self.params]

		dWe = theano.shared(self.We.get_value() * 0)
		gWe = T.grad(cost, self.We)
		dWe_update = mu*dWe - lr*gWe
		We_update = self.We + dWe_update
		if normalize:
			We_update /= We_update.norm(2)

		updates = [
			(p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
		] + [
			(dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
		] + [
			(self.We, We_update), (dWe, dWe_update)
		]

		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates,
			allow_input_downcast=True,
		)

		train_debug_op = theano.function(
			inputs=[thX, thY],
			outputs=[cost, prediction],
			updates=updates,
			allow_input_downcast=True,
		)

		costs = []
		for i in range(epochs):
			if debug:
				n_correct = 0
				n_total = 0
				cost = 0

			X = shuffle(X)
			for j in range(N):
				if np.random.random() < 0.01 or len(X[j]) < 2:
					input_sequence = [0] + X[j]
					output_sequence = X[j] + [1]
				else:
					input_sequence = [0] + X[j][:-1]
					output_sequence = X[j]
				if debug:
					n_total += len(output_sequence)

				# test:

				try:
					# we set 0 to start and 1 to end
					if debug:
						c, p = train_debug_op(input_sequence, output_sequence)
					else:
						train_op(input_sequence, output_sequence)
				except Exception as e:
					PY_X, pred = self.predict_op(input_sequence)
					print('input_sequence len: %d' % len(input_sequence))
					print('PY_X.shape:', PY_X.shape)
					print('pred.shape:', pred.shape)
					raise e

				if debug:
					cost += c
					n_correct += np.sum(np.array(output_sequence) == np.array(p))
					if j % print_period == 0:
						score = float(n_correct) / n_total
						print('j/N: %d/%d correct rate so far: %.6f%%\r' % (j, N, score*100))
						# sys.stdout.write('j/N: %d/%d correct rate so far: %.6f%%\r' % (j, N, score*100))
						# sys.stdout.flush()
			if debug:
				costs.append(cost)
				score = float(n_correct) / n_total
				print('epoch: %d, cost: %.6f, correct rate: %.6f%%' % (i, cost, score*100))

		if debug:
			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()


def train_wikipedia(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', RecurrentUnit=GRU, is_wikipedia=True, debug=False):
	if is_wikipedia:
		sentences, word2idx = get_wikipedia_data(n_files=10, n_vocab=2000)
	else:
		sentences, word2idx = get_sentences_with_word2idx_limit_vocab()

	if debug:
		from datetime import datetime
		print('\bFinished retriving data...')
		print('vocab size: %d, number of sentences: %d\n' % (len(word2idx), len(sentences)))
		t0 = datetime.now()

	rnn = RNN(30, [30], len(word2idx), RecurrentUnit=RecurrentUnit, activation=T.nnet.relu)
	rnn.fit(sentences, epochs=10, learning_rate=1e-4, normalize=True, debug=debug)

	if debug:
		print('Done training! Elapsed time: %s' % (datetime.now() - t0))
		print('Now begin to save the files...\n')

	np.save(we_file, rnn.We.get_value())
	with open(w2i_file, 'w') as f:
		json.dump(word2idx, f)

	if debug:
		print('Successfully save the file!\n')


def generate_wikipedia():
	# TODO: implement a function to generate a wikipedia article
	pass


def find_analogy(w1, w2, w3, we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json'):
	We = np.load(we_file)
	with open(w2i_file) as f:
		word2idx = json.load(f)

	king = We[word2idx[w1]]
	man = We[word2idx[w2]]
	woman = We[word2idx[w3]]
	v0 = king - man + woman

	def dist1(a, b):
		return np.linalg.norm(a - b)
	def dist2(a, b):
		return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

	for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
		min_dist = float('inf')
		best_word = ''
		for word, idx in word2idx.items():
			if word not in (w1, w2, w3):
				v1 = We[idx]
				d = dist(v0, v1)
				if d < min_dist:
					min_dist = d
					best_word = word
		print('closest match by %s distance: %s' % (name, word))
		print('%s - %s = %s - %s' % (w1, w2, best_word, w3))


if __name__ == '__main__':
	is_wikipedia = False
	is_gru = False
	is_training = False

	if is_wikipedia:
		we  = 'wikipedia_word_embeddings.npy'
		w2i = 'wikipedia_word2idx.json'
	else:
		we  = 'brown_word_embeddings.npy'
		w2i = 'brown_word2idx.json'

	if is_gru:
		we  = 'gru_' + we
		w2i = 'gru_' + w2i
		if is_training:
			train_wikipedia(we, w2i, RecurrentUnit=GRU, is_wikipedia=is_wikipedia, debug=True)
	else:
		we  = 'lstm_' + we
		w2i = 'lstm_' + w2i
		if is_training:
			train_wikipedia(we, w2i, RecurrentUnit=LSTM, is_wikipedia=is_wikipedia, debug=True)

	find_analogy('king', 'man', 'woman', we, w2i)
	find_analogy('france', 'paris', 'london', we, w2i)
	find_analogy('france', 'paris', 'rome', we, w2i)
	find_analogy('paris', 'france', 'italy', we, w2i)

