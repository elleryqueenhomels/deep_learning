# Using a Simple RNN to classify different poetry styles
# Simple Recurrent Unit / Elman Unit

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_activation, get_poetry_classifier_data


class SimpleRNN(object):
	def __init__(self, M, V, activation_type=1):
		self.M = M # hidden layer size
		self.V = V # vocabulary size
		self.activation = get_activation(activation_type)

	def fit(self, X, Y, epochs=500, learning_rate=1e-4, decay = 0.9999, momentum=0.99, debug=False, Xvalid=None, Yvalid=None):
		mu = momentum

		# X: (N, T(n)), Y: (N, )
		N = len(X)
		M = self.M
		V = self.V
		K = len(set(Y))

		debug = debug and (Xvalid is not None) and (Yvalid is not None) and (len(Xvalid) == len(Yvalid))

		# initialize weights
		Wx = init_weight(V, M)
		Wh = init_weight(M, M)
		bh = np.zeros(M)
		h0 = np.zeros(M)
		Wo = init_weight(M, K)
		bo = np.zeros(K)

		thX, thY, py_x, prediction = self.set(Wx, Wh, bh, h0, Wo, bo)

		cost = -T.mean(T.log(py_x[thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
		lr = T.scalar('learning_rate')

		# momentum only
		updates = []
		for p, dp, g in zip(self.params, dparams, grads):
			updates += [
				(p, p + mu*dp - lr*g),
				(dp, mu*dp - lr*g)
			]

		cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction], allow_input_downcast=True)

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

		if debug:
			costs = []
			Nvalid = len(Yvalid)
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			if debug:
				n_correct = 0
				cost = 0

			for j in range(N):
				if debug:
					c, p = train_debug_op(X[j], Y[j], learning_rate)
					cost += c
					if p == Y[j]:
						n_correct += 1
				else:
					train_op(X[j], Y[j], learning_rate)
				learning_rate *= decay

			if debug:
				n_correct_valid = 0
				for j in range(Nvalid):
					p = self.predict_op(Xvalid[j])
					if p == Yvalid[j]:
						n_correct_valid += 1
				costs.append(cost)
				score = n_correct / N
				score_valid = n_correct_valid / Nvalid
				print('epoch=%d, cost=%.6f, score=%.6f%%' % (i, cost, score*100))
				print('validation score=%.6f%%' % (score_valid*100))

		if debug:
			cost, n_correct, n_correct_valid = 0, 0, 0
			for j in range(N):
				c, p = cost_predict_op(X[j], Y[j])
				cost += c
				if p == Y[j]:
					n_correct += 1
			for j in range(Nvalid):
				p = self.predict_op(Xvalid[j])
				if p == Yvalid[j]:
					n_correct_valid += 1
			score = n_correct / N
			score_valid = n_correct_valid / Nvalid
			print('Finally: cost=%.6f, score=%.6f%%' % (cost, score*100))
			print('Finally: validation score=%.6f%%' % (score_valid*100))

			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()

	def set(self, Wx, Wh, bh, h0, Wo, bo):
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		thX = T.ivector('X') # thX: (T, )
		thY = T.iscalar('Y') # thY: scalar

		def recurrence(x_t, h_t1):
			# returns h(t), y(t)
			# x_t: scalar, h_t: (M, ), y_t: (1, K)
			h_t = self.activation(self.Wx[x_t] + h_t1.dot(self.Wh) + self.bh)
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h, y], _ = theano.scan(
			fn=recurrence,
			sequences=thX,
			n_steps=thX.shape[0],
			outputs_info=[self.h0, None]
		)

		# y: (T, 1, K), h: (T, M)
		py_x = y[-1, 0, :] # py_x: (K, )
		prediction = T.argmax(py_x) # prediction: scalar
		self.predict_op = theano.function(
			inputs=[thX],
			outputs=prediction,
			allow_input_downcast=True
		)

		return thX, thY, py_x, prediction

	def save(self, filename):
		array = [p.get_value() for p in self.params]
		np.savez(filename, *array)

	@staticmethod
	def load(filename, activation_type=1):
		npz = np.load(filename)
		Wx = npz['arr_0']
		Wh = npz['arr_1']
		bh = npz['arr_2']
		h0 = npz['arr_3']
		Wo = npz['arr_4']
		bo = npz['arr_5']
		V, M = Wx.shape
		rnn = SimpleRNN(M, V, activation_type)
		rnn.set(Wx, Wh, bh, h0, Wo, bo)
		return rnn


def train_poetry():
	X, Y, V = get_poetry_classifier_data(load_cached=False)
	print('type(X):', type(X), 'type(Y):', type(Y))
	print('\nV=%d, len(X)=%d\n' % (V, len(X)))
	X, Y = shuffle(X, Y)
	Nvalid = 100
	Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:]
	X, Y = X[:-Nvalid], Y[:-Nvalid]
	rnn = SimpleRNN(30, V, activation_type=1)
	rnn.fit(X, Y, epochs=1000, learning_rate=1e-4, decay=0.999999, debug=True, Xvalid=Xvalid, Yvalid=Yvalid)


if __name__ == '__main__':
	train_poetry()

