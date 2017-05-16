# Using a Simple RNN to model language - generate poetry
# Simple Recurrent Unit / Elman Unit

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_activation, get_robert_frost


class SimpleRNN(object):
	def __init__(self, D, M, V, activation_type=1):
		self.D = D # dimensionality of word vector in word embedding
		self.M = M # hidden layer size
		self.V = V # vocabulary size
		self.activation = get_activation(activation_type)

	def fit(self, X, epochs=500, learning_rate=1e-4, momentum=0.99, debug=False):
		# Use float32 for Theano GPU mode
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)

		# X: (N, T(n))
		N = len(X)
		D = self.D
		M = self.M
		V = self.V

		# initialize weigths
		We = init_weight(V, D).astype(np.float32)
		Wx = init_weight(D, M).astype(np.float32)
		Wh = init_weight(M, M).astype(np.float32)
		bh = np.zeros(M).astype(np.float32)
		h0 = np.zeros(M).astype(np.float32)
		Wo = init_weight(M, V).astype(np.float32)
		bo = np.zeros(V).astype(np.float32)

		# make them theano shared
		self.We = theano.shared(We)
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		thX = T.ivector('X[n]') # thX: (T, ), here 'T' represents 'T(n)'
		Ei = self.We[thX] # Ei: (T, D)
		thY = T.ivector('Y') # thY: (T, ), corresponding to thX

		# sentence input:
		# [START, w1, w2, ... , wn]
		# sentence target:
		# [w1, w2, ... , wn, END]

		def recurrence(x_t, h_t1):
			# returns h(t), y(t)
			# x_t: (D, ), h_t: (M, ), y_t: (1, V)
			h_t = self.activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h, y], _ = theano.scan(
			fn=recurrence,
			sequences=Ei,
			n_steps=Ei.shape[0],
			outputs_info=[self.h0, None]
		)

		# y: (T, 1, V), h: (T, M)
		py_x = y[:, 0, :] # py_x: (T, V)
		prediction = T.argmax(py_x, axis=1) # prediction: (T, )

		cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		# momentum only
		updates = []
		for p, dp, g in zip(self.params, dparams, grads):
			updates += [
				(p, p + mu*dp - lr*g),
				(dp, mu*dp - lr*g)
			]

		self.predict_op = theano.function(inputs=[thX], outputs=prediction, allow_input_downcast=True)
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates
		)

		train_debug_op = theano.function(
			inputs=[thX, thY],
			outputs=[cost, prediction],
			updates=updates
		)

		if debug:
			costs = []
			n_total = sum((len(sentence) + 1) for sentence in X)
		for i in range(epochs):
			X = shuffle(X)
			if debug:
				n_correct = 0
				cost = 0

			for j in range(N):
				input_sequence = [0] + X[j]
				output_sequence = X[j] + [1]

				if debug:
					c, p = train_debug_op(input_sequence, output_sequence)
					cost += c
					n_correct += np.sum(np.array(p) == np.array(output_sequence))
				else:
					train_op(input_sequence, output_sequence)

			if debug:
				costs.append(cost)
				score = n_correct / n_total
				print('epoch=%d, cost=%.6f, score=%.6f%%' % (i, cost, score*100))

		if debug:
			n_correct, cost = 0, 0
			for j in range(N):
				input_sequence = [0] + X[j]
				output_sequence = X[j] + [1]
				c, p = cost_predict_op(input_sequence, output_sequence)
				cost += c
				n_correct += np.sum(np.array(p) == np.array(output_sequence))
			score = n_correct / n_total
			print('\nFinally: cost=%.6f, score=%.6f%%\n' % (cost, score*100))

			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()

	def save(self, filename):
		array = [p.get_value() for p in self.params]
		np.savez(filename, *array)

	@staticmethod
	def load(filename, activation_type=1):
		npz = np.load(filename)
		We = npz['arr_0']
		Wx = npz['arr_1']
		Wh = npz['arr_2']
		bh = npz['arr_3']
		h0 = npz['arr_4']
		Wo = npz['arr_5']
		bo = npz['arr_6']
		V, D = We.shape
		_, M = Wx.shape
		rnn = SimpleRNN(D, M, V, activation_type)
		rnn.set(We, Wx, Wh, bh, h0, Wo, bo)
		return rnn

	def set(self, We, Wx, Wh, bh, h0, Wo, bo):
		self.We = theano.shared(We)
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		thX = T.ivector('X[n]') # thX: (T, ), here 'T' represents 'T(n)'
		Ei = self.We[thX] # Ei: (T, D)

		def recurrence(x_t, h_t1):
			# returns h(t), y(t)
			# x_t: (D, ), h_t: (M, ), y_t: (1, V)
			h_t = self.activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h, y], _ = theano.scan(
			fn=recurrence,
			sequences=Ei,
			n_steps=Ei.shape[0],
			outputs_info=[self.h0, None]
		)

		# y: (T, 1, V), h: (T, M)
		py_x = y[:, 0, :] # py_x: (T, V)
		prediction = T.argmax(py_x, axis=1) # prediction: (T, )
		self.predict_op = theano.function(
			inputs=[thX],
			outputs=prediction,
			allow_input_downcast=True
		)

	def generate(self, pi, word2idx, lines=4, print_out=True):
		# convert word2idx -> idx2word
		idx2word = {v:k for k, v in word2idx.items()}
		V = len(pi)

		n_lines = 0
		sentences = []

		# Why? because using the START symbol will always yield the same first word!
		X = [np.random.choice(V, p=pi)]
		sentence = [idx2word[X[0]]]

		while n_lines < lines:
			P = self.predict_op(X)[-1]
			X.append(P)
			if P > 1:
				word = idx2word[P]
				sentence.append(word)
			elif P == 1:
				n_lines += 1
				# sentence.append('\n')
				sentences.append(' '.join(sentence))
				if n_lines < lines:
					X = [np.random.choice(V, p=pi)]
					sentence = [idx2word[X[0]]]

		if print_out:
			for sentence in sentences:
				print(sentence)

		return sentences


def train_poetry():
	sentences, word2idx = get_robert_frost()
	rnn = SimpleRNN(30, 30, len(word2idx), activation_type=1)
	rnn.fit(sentences, epochs=2000, learning_rate=1e-4, debug=True)
	rnn.save('RNN_D30_M30_epochs2000_relu.npz')

def generate_poetry():
	sentences, word2idx = get_robert_frost()
	rnn = SimpleRNN.load('RNN_D30_M30_epochs2000_relu.npz', activation_type=1)

	# determine initial state distribution for starting words
	V = len(word2idx)
	pi = np.zeros(V)
	for sentence in sentences:
		pi[sentence[0]] += 1
	pi /= pi.sum()

	rnn.generate(pi, word2idx)


if __name__ == '__main__':
	train_poetry()
	generate_poetry()

