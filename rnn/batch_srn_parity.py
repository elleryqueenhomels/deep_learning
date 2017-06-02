# Using a Simple RNN to solve Parity Problem with Batch Training
# Simple Recurrent Unit / Elman Unit

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN(object):
	def __init__(self, M, activation=T.nnet.relu):
		self.M = M # hidden layer size
		self.f = activation

	def fit(self, X, Y, epochs=100, batch_sz=20, learning_rate=1e-1, momentum=0.99, debug=False, print_period=10):
		# for convenience
		lr = learning_rate
		mu = momentum
		D = X[0].shape[1] # X: (N, T, D)
		K = len(set(Y.flatten()))
		N = len(Y)
		M = self.M

		# initialize weights
		Wx = init_weight(D, M)
		Wh = init_weight(M, M)
		bh = np.zeros(M)
		h0 = np.zeros(M)
		Wo = init_weight(M, K)
		bo = np.zeros(K)

		# make them theano shared
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)
		self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		thX = T.fmatrix('X') # will represent multiple batches concatenated
		thY = T.ivector('Y')
		thStartPoints = T.ivector('start_points')

		XW = thX.dot(self.Wx)

		def recurrence(xw_t, is_start, h_t1, h0):
			# if at a boundary, state should be h0
			h_t = T.switch(
				T.eq(is_start, 1),
				self.f(xw_t + h0.dot(self.Wh) + self.bh),
				self.f(xw_t + h_t1.dot(self.Wh) + self.bh)
			)
			return h_t

		h, _ = theano.scan(
			fn=recurrence,
			sequences=[XW, thStartPoints],
			n_steps=XW.shape[0],
			outputs_info=[self.h0],
			non_sequences=[self.h0],
		)

		# h: (batch_sz * T, M), py_x: (batch_sz * T, K)
		py_x = T.nnet.softmax(h.dot(self.Wo) + self.bo)
		prediction = T.argmax(py_x, axis=1) # prediction: (batch_sz * T, )

		cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
		grads = T.grad(cost, self.params)
		dparams = [theano.shared(p.get_value() * 0) for p in self.params]

		updates = [
			(p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
		] + [
			(dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
		]

		train_op = theano.function(
			inputs=[thX, thY, thStartPoints],
			updates=updates,
			allow_input_downcast=True,
		)

		train_debug_op = theano.function(
			inputs=[thX, thY, thStartPoints],
			outputs=[cost, prediction, py_x],
			updates=updates,
			allow_input_downcast=True,
		)

		costs = []
		n_batches = N // batch_sz
		sequenceLength = X.shape[1]

		# if each sequence was of variable length, we would need to
		# initialize this inside the loop for every new batch
		startPoints = np.zeros(sequenceLength * batch_sz, dtype=np.int32)
		for b in range(batch_sz):
			startPoints[b * sequenceLength] = 1
		for i in range(epochs):
			X, Y = shuffle(X, Y)
			if debug:
				n_correct = 0
				cost = 0

			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)].reshape(sequenceLength * batch_sz, D)
				Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)].reshape(sequenceLength * batch_sz).astype(np.int32)

				if debug:
					c, p, out = train_debug_op(Xbatch, Ybatch, startPoints)
					cost += c
					for b in range(batch_sz):
						idx = sequenceLength * (b + 1) - 1
						if p[idx] == Ybatch[idx]:
							n_correct += 1
				else:
					train_op(Xbatch, Ybatch, startPoints)

			if debug:
				costs.append(cost)
				score = float(n_correct) / N
				print('py_x.shape:', out.shape)
				print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, cost, score*100))
			if n_correct == N:
				score = float(n_correct) / N
				print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, cost, score*100))
				break

		if debug:
			plt.plot(costs)
			plt.title('Cross-Entropy Costs')
			plt.show()


def parity(nbit=12, learning_rate=1e-3, epochs=50):
	X, Y = all_parity_pairs_with_sequence_labels(nbit)

	rnn = SimpleRNN(4, activation=T.nnet.relu)
	rnn.fit(X, Y, epochs=epochs, batch_sz=10, learning_rate=learning_rate, debug=True)


if __name__ == '__main__':
	parity()

