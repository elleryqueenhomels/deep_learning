import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN(object):
	def __init__(self, M, activation=tf.nn.relu):
		self.M = M # hidden layer size
		self.f = activation

	def fit(self, X, Y, epochs=100, learning_rate=1e-3, momentum=0.99, debug=False, print_period=10):
		# for convenience
		lr = learning_rate
		mu = momentum

		N, T, D = X.shape
		K = len(set(Y.flatten()))
		M = self.M

		# initialize weights
		Wx = init_weight(D, M).astype(np.float32)
		Wh = init_weight(M, M).astype(np.float32)
		bh = np.zeros(M, dtype=np.float32)
		h0 = np.zeros(M, dtype=np.float32)
		Wo = init_weight(M, K).astype(np.float32)
		bo = np.zeros(K, dtype=np.float32)

		# make them tensorflow vars
		self.Wx = tf.Variable(Wx)
		self.Wh = tf.Variable(Wh)
		self.bh = tf.Variable(bh)
		self.h0 = tf.Variable(h0)
		self.Wo = tf.Variable(Wo)
		self.bo = tf.Variable(bo)

		tfX = tf.placeholder(tf.float32, shape=(T, D), name='X')
		tfY = tf.placeholder(tf.int32,   shape=(T,  ), name='Y')

		XWx = tf.matmul(tfX, self.Wx)

		def recurrence(h_t1, xwx_t):
			# tf.matmul() only works with 2-D objects
			# we want to return a 1-D object of size M
			# so that the final result is (T, M)
			# not (T, 1, M)
			h_t = self.f(xwx_t + tf.matmul(tf.reshape(h_t1, [1, M]), self.Wh) + self.bh)
			return tf.reshape(h_t, [M, ])

		h = tf.scan(
			fn=recurrence,
			elems=XWx,
			initializer=self.h0
		)

		logits = tf.matmul(h, self.Wo) + self.bo

		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=tfY
			)
		)

		predict_op = tf.argmax(logits, axis=1)

		train_op = tf.train.AdamOptimizer(lr).minimize(cost)

		init = tf.global_variables_initializer()

		costs = []
		with tf.Session() as sess:
			sess.run(init)

			for i in range(epochs):
				X, Y = shuffle(X, Y)
				if debug:
					batch_cost = 0
					n_correct = 0

				for j in range(N):
					if debug:
						ops = [train_op, cost, predict_op]
						_, c, p = sess.run(ops, feed_dict={tfX: X[j], tfY: Y[j]})
						batch_cost += c
						n_correct += (p[-1] == Y[j, -1])
					else:
						sess.run(train_op, feed_dict={tfX: X[j], tfY: Y[j]})

				if debug:
					costs.append(batch_cost)
					score = float(n_correct) / N
					print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, batch_cost, score*100))
					if n_correct == N:
						break

		if debug:
			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()


def parity(nbit=12, learning_rate=1e-3, epochs=50):
	X, Y = all_parity_pairs_with_sequence_labels(nbit)
	X = X.astype(np.float32)

	rnn = SimpleRNN(4, activation=tf.nn.relu)
	rnn.fit(X, Y, epochs=epochs, learning_rate=learning_rate, debug=True)


if __name__ == '__main__':
	parity()

