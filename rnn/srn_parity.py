# Using a Simple RNN to solve the Parity Problem
# Simple Recurrent Unit / Elman Unit

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_activation, all_parity_pairs_with_sequence_labels


class SimpleRNN(object):
	def __init__(self, M, activation_type=1):
		self.M = M # hidden layer size
		self.activation = get_activation(activation_type)

	def fit(self, X, Y, epochs=100, learning_rate=1e-4, momentum=0.99, debug=False):
		# Use float32 for Theano GPU mode
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		X = X.astype(np.float32)
		Y = Y.astype(np.int32)

		# X: (N, T(n), D), Y: (N, T(n))
		D = X[0].shape[1]
		K = len(set(Y.flatten()))
		N = len(Y)
		M = self.M # just for convinience

		# initialize weights
		Wx = init_weight(D, M).astype(np.float32)
		Wh = init_weight(M, M).astype(np.float32)
		bh = np.zeros(M, dtype=np.float32)
		h0 = np.zeros(M, dtype=np.float32)
		Wo = init_weight(M, K).astype(np.float32)
		bo = np.zeros(K, dtype=np.float32)

		# make them theano shared
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		thX = T.fmatrix('X[n]') # thX: (T, D), here 'T' represents 'T(n)'
		thY = T.ivector('Y[n]') # thY: (T,  ), here 'T' represents 'T(n)'

		def recurrence(x_t, h_t1):
			# returns h_t, y_t
			# x_t: (D, ), h_t: (M, ), y_t: (1, K)
			h_t = self.activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h, y], _ = theano.scan(
			fn=recurrence,
			sequences=thX,
			n_steps=thX.shape[0],
			outputs_info=[self.h0, None]
		)

		# y: (T, 1, K), h: (T, M)
		py_x = y[:, 0, :] # py_x: (T, K)
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

		self.predict_op = theano.function(inputs=[thX], outputs=prediction)
		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

		train_op = theano.function(
			inputs=[thX, thY],
			updates=updates
		)
		train_debug_op = theano.function(
			inputs=[thX, thY],
			outputs=[cost, prediction, y],
			updates=updates
		)

		costs = []
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			if debug:
				n_correct = 0
				cost = 0

			for j in range(N):
				if debug:
					c, p, out = train_debug_op(X[j], Y[j])
					cost += c
					if p[-1] == Y[j,-1]:
						n_correct += 1
				else:
					train_op(X[j], Y[j])

			if debug:
				score = n_correct / N
				costs.append(cost)
				print('y.shape:', out.shape)
				print('epoch=%d, cost=%.6f, score=%.6f%%' % (i, cost, score*100))

		if debug:
			n_correct, cost = 0, 0
			for n in range(N):
				c, p = cost_predict_op(X[n], Y[n])
				cost += c
				if p[-1] == Y[n,-1]:
					n_correct += 1
			score = n_correct / N
			print('\nFinally: cost=%.6f, score=%.6f%%\n' % (cost, score*100))

			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()


def parity(nbit=12, learning_rate=1e-4, epochs=20, debug=True):
	X, Y = all_parity_pairs_with_sequence_labels(nbit)

	rnn = SimpleRNN(4, activation_type=1)
	rnn.fit(X, Y, epochs=epochs, learning_rate=learning_rate, debug=debug)


if __name__ == '__main__':
	parity()

