# Using a Rated RNN to model language - generate poetry
# Rated Recurrent Unit

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_activation, get_robert_frost


class RRNN(object):
	def __init__(self, D, M, V, activation_type=1):
		self.D = D # dimensionality of word vector in word embedding
		self.M = M # hidden layer size
		self.V = V # vocabulary size
		self.activation = get_activation(activation_type)

	def fit(self, X, epochs=500, learning_rate=1e-4, momentum=0.99, debug=False):
		mu = momentum

		# X: (N, T(n))
		N = len(X)
		D = self.D
		M = self.M
		V = self.V

		# initialize weigths
		We = init_weight(V, D)
		Wx = init_weight(D, M)
		Wh = init_weight(M, M)
		bh = np.zeros(M)
		h0 = np.zeros(M)
		# z = np.ones(M)
		Wxz = init_weight(D, M)
		Whz = init_weight(M, M)
		bz = np.zeros(M)
		Wo = init_weight(M, V)
		bo = np.zeros(V)

		thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo)

		lr = T.scalar('learning_rate')

		cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

		# momentum only
		updates = []
		for p, dp, g in zip(self.params, dparams, grads):
			updates += [
				(p, p + mu*dp - lr*g),
				(dp, mu*dp - lr*g)
			]

		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		train_op = theano.function(
			inputs=[thX, thY, lr],
			updates=updates,
			allow_input_downcast=True
		)

		train_debug_op = theano.function(
			inputs=[thX, thY, lr],
			outputs=[cost, prediction],
			updates=updates,
			allow_input_downcast=True
		)

		costs = []
		for i in range(epochs):
			X = shuffle(X)
			if debug:
				n_correct = 0
				n_total = 0
				cost = 0

			for j in range(N):
				if np.random.random() < 0.1:
					input_sequence = [0] + X[j]
					output_sequence = X[j] + [1]
				else:
					input_sequence = [0] + X[j][:-1]
					output_sequence = X[j]

				if debug:
					c, p = train_debug_op(input_sequence, output_sequence, learning_rate)
					cost += c
					n_correct += np.sum(np.array(p) == np.array(output_sequence))
					n_total += len(output_sequence)
				else:
					train_op(input_sequence, output_sequence, learning_rate)

				if (i + 1) % 500 == 0:
					learning_rate /= 2

			if debug:
				costs.append(cost)
				score = n_correct / n_total
				print('epoch=%d, cost=%.6f, score=%.6f%%' % (i, cost, score*100))

		if debug:
			n_correct, n_total, cost = 0, 0, 0
			for j in range(N):
				input_sequence = [0] + X[j][:-1]
				output_sequence = X[j]
				c, p = cost_predict_op(input_sequence, output_sequence)
				cost += c
				n_correct += np.sum(np.array(p) == np.array(output_sequence))
				n_total += len(output_sequence)
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
		Wxz = npz['arr_5']
		Whz = npz['arr_6']
		bz = npz['arr_7']
		Wo = npz['arr_8']
		bo = npz['arr_9']
		V, D = We.shape
		_, M = Wx.shape
		rnn = RRNN(D, M, V, activation_type)
		rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo)
		return rnn

	def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo):
		self.We = theano.shared(We)
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wxz = theano.shared(Wxz)
		self.Whz = theano.shared(Whz)
		self.bz = theano.shared(bz)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

		thX = T.ivector('X[n]') # thX: (T, ), here 'T' represents 'T(n)'
		Ei = self.We[thX] # Ei: (T, D)
		thY = T.ivector('Y') # thY: (T, ), corresponding to thX

		def recurrence(x_t, h_t1):
			# returns h(t), y(t)
			# x_t: (D, ), h_t: (M, ), y_t: (1, V)
			hhat_t = self.activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz) # z(t) is like a "gate", depending on x(t) and h(t-1)
			h_t = (1 - z_t) * h_t1 + z_t * hhat_t # just like a Low Pass Filter
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
			outputs=[py_x, prediction],
			allow_input_downcast=True
		)
		return thX, thY, py_x, prediction

	def generate(self, word2idx, lines=4, print_out=True):
		# convert word2idx -> idx2word
		idx2word = {v:k for k, v in word2idx.items()}
		V = len(word2idx)

		n_lines = 0
		sentences = []

		# Why? because using the START symbol will always yield the same first word!
		X = [0]
		sentence = []

		while n_lines < lines:
			PY_X, _ = self.predict_op(X)
			PY_X = PY_X[-1]
			P = np.random.choice(V, p=PY_X)
			X.append(P)
			if P > 1:
				word = idx2word[P]
				sentence.append(word)
			elif P == 1:
				n_lines += 1
				# sentence.append('\n')
				sentences.append(' '.join(sentence))
				if n_lines < lines:
					X = [0]
					sentence = []

		if print_out:
			for sentence in sentences:
				print(sentence)

		return sentences


def train_poetry():
	sentences, word2idx = get_robert_frost()
	rnn = RRNN(50, 50, len(word2idx), activation_type=1)
	rnn.fit(sentences, epochs=2000, learning_rate=1e-3, debug=True)
	rnn.save('RRNN_D50_M50_epochs2000_relu.npz')

def generate_poetry():
	sentences, word2idx = get_robert_frost()
	rnn = RRNN.load('RRNN_D50_M50_epochs2000_relu.npz', activation_type=1)
	rnn.generate(word2idx, lines=4, print_out=True)


if __name__ == '__main__':
	train_poetry()
	generate_poetry()

