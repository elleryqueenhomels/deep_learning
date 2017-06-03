import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from tensorflow.python.ops import rnn as rnn_module

######## This only works for pre-1.0 versions ########
# from tensorflow.python.ops.rnn import rnn as get_rnn_output
# from tensorflow.python.ops.rnn_cell import BasicRNNCell, GRUCell
######################################################

######## This works for TensorFlow v1.0 ##############
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell, LSTMCell
######################################################

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


def x2sequence(x, T, D, batch_sz):
	# Permuting batch_size and n_steps
	x = tf.transpose(x, [1, 0, 2])
	# Reshaping to (n_steps * batch_sz, n_input)
	x = tf.reshape(x, [T * batch_sz, D])
	# Split to get a list of 'n_steps' tensors of shape (batch_sz, n_input)
	# x = tf.split(0, T, x) # v0.1
	x = tf.split(x, T) # v1.0
	return x


class SimpleRNN(object):
	def __init__(self, M, activation=tf.nn.relu, RecurrentUnit=BasicRNNCell):
		self.M = M # hidden layer sizes
		self.f = activation
		self.RecurrentUnit = RecurrentUnit

	def fit(self, X, Y, epochs=100, batch_sz=20, learning_rate=1e-1, momentum=0.99, debug=True, print_period=10):
		# for convenience
		lr = learning_rate
		mu = momentum
		activation = self.f
		RecurrentUnit = self.RecurrentUnit

		N, T, D = X.shape # X is of size: (N, T, D)
		K = len(set(Y.flatten()))
		M = self.M

		# initial weights
		# note: Wx, Wh, bh are all part of the RNN unit and will be created
		# 		by BasicRNNCell
		Wo = init_weight(M, K).astype(np.float32)
		bo = np.zeros(K, dtype=np.float32)

		# make them tf variables
		self.Wo = tf.Variable(Wo)
		self.bo = tf.Variable(bo)

		# tf Graph input
		tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
		tfY = tf.placeholder(tf.int32, shape=(batch_sz, T), name='targets')

		# turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
		sequenceX = x2sequence(tfX, T, D, batch_sz)

		# create the Simple Recurrent Unit (Elman Unit)
		rnn_unit = RecurrentUnit(num_units=M, activation=activation)

		# Get RNN cell output
		# outputs, states = rnn_module.rnn(rnn_unit, sequenceX, dtype=tf.float32)
		outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

		# outputs are now of size (T, batch_sz, M)
		# so make it (batch_sz, T, M)
		outputs = tf.transpose(outputs, [1, 0, 2])
		outputs = tf.reshape(outputs, [T * batch_sz, M])

		# Linear activation, using RNN inner loop last output
		logits = tf.matmul(outputs, self.Wo) + self.bo
		predict_op = tf.argmax(logits, 1)
		targets = tf.reshape(tfY, [T * batch_sz, ])

		cost_op = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=targets
			)
		)

		# train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost_op)
		train_op = tf.train.AdamOptimizer(lr).minimize(cost_op)

		costs = []
		n_batches = N // batch_sz

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for i in range(epochs):
				X, Y = shuffle(X, Y)
				if debug:
					cost = 0
					n_correct = 0

				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
					Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

					if debug:
						ops = [train_op, cost_op, predict_op]
						_, c, p = sess.run(ops, feed_dict={tfX: Xbatch, tfY: Ybatch})
						cost += c
						for b in range(batch_sz):
							idx = (b + 1) * T - 1
							n_correct += (p[idx] == Ybatch[b][-1])
					else:
						sess.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

				if debug:
					costs.append(cost)
					if n_correct == N:
						score = float(n_correct) / N
						print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, cost, score*100))
						break
					if i % print_period == 0:
						score = float(n_correct) / N
						print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, cost, score*100))

		if debug:
			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()


def parity(nbit=12, learning_rate=1e-3, epochs=50):
	X, Y = all_parity_pairs_with_sequence_labels(nbit)

	rnn = SimpleRNN(4, activation=tf.nn.relu, RecurrentUnit=LSTMCell)
	rnn.fit(
		X, Y,
		epochs=epochs,
		batch_sz=10,
		learning_rate=learning_rate,
		debug=True
	)


if __name__ == '__main__':
	parity()

