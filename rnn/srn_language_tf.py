import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost


class SimpleRNN(object):
	def __init__(self, D, M, V, session, activation=tf.nn.relu):
		self.D = D # dimensionality of word embeddings
		self.M = M # hidden layer size
		self.V = V # vocabulary size
		self.f = activation
		self.session = session

	def set_session(self, session):
		self.session = session

	def build(self, We, Wx, Wh, bh, h0, Wo, bo):
		# make them TF variables
		self.We = tf.Variable(We)
		self.Wx = tf.Variable(Wx)
		self.Wh = tf.Variable(Wh)
		self.bh = tf.Variable(bh)
		self.h0 = tf.Variable(h0)
		self.Wo = tf.Variable(Wo)
		self.bo = tf.Variable(bo)
		self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

		# for convenience
		D = self.D
		M = self.M
		V = self.V

		# placeholders
		self.tfX = tf.placeholder(tf.int32, shape=(None, ), name='X')
		self.tfY = tf.placeholder(tf.int32, shape=(None, ), name='Y')

		# convert word indexes to word vectors
		# this would be equivalent to do
		# We[thX] in NumPy / Theano
		# or:
		# X_one_hot = one_hot_encode(X)
		# X_one_hot.dot(We)
		XWe = tf.nn.embedding_lookup(self.We, self.tfX)
		XWx = tf.matmul(XWe, self.Wx)

		def recurrence(h_t1, xwx_t):
			h_t1 = tf.reshape(h_t1, [1, M])
			h_t  = self.f(xwx_t + tf.matmul(h_t1, self.Wh) + self.bh)
			h_t  = tf.reshape(h_t, [M, ])
			return h_t

		h = tf.scan(
			fn=recurrence,
			elems=XWx,
			initializer=self.h0
		)

		# output
		logits = tf.matmul(h, self.Wo) + self.bo
		self.predict_op = tf.argmax(logits, axis=1)
		self.output_probs = tf.nn.softmax(logits)

		nce_weights = tf.transpose(self.Wo, [1, 0]) # need to be (V, M), not (M, V)
		nce_biases  = self.bo

		h = tf.reshape(h, [-1, M])
		labels = tf.reshape(self.tfY, [-1, 1])

		self.cost = tf.reduce_mean(
			tf.nn.sampled_softmax_loss(
				weights=nce_weights,
				biases=nce_biases,
				labels=labels,
				inputs=h,
				num_sampled=50, # number of negative samples
				num_classes=V
			)
		)

		self.session.run(tf.global_variables_initializer())

	def fit(self, X, epochs=100, learning_rate=1e-3, momentum=0.99, debug=False):
		# for convenience
		lr = learning_rate
		mu = momentum

		N = len(X)
		D = self.D
		M = self.M
		V = self.V
		sess = self.session

		# initialize weights
		We = init_weight(V, D).astype(np.float32)
		Wx = init_weight(D, M).astype(np.float32)
		Wh = init_weight(M, M).astype(np.float32)
		bh = np.zeros(M, dtype=np.float32)
		h0 = np.zeros(M, dtype=np.float32)
		Wo = init_weight(M, V).astype(np.float32)
		bo = np.zeros(V, dtype=np.float32)

		# build tensorflow functions
		self.build(We, Wx, Wh, bh, h0, Wo, bo)

		train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
		# train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(self.cost)

		# initialize all variables
		sess.run(tf.global_variables_initializer())

		# sentence input
		# [START, w1, w2, ... , wn]
		# sentence target
		# [w1,    w2, w3, ... , END]

		costs = []
		for i in range(epochs):
			X = shuffle(X)
			if debug:
				cost = 0
				n_total = 0
				n_correct = 0

			for j in range(N):
				if np.random.random() < 0.1 or len(X[j]) < 2:
					input_sequence  = [0] + X[j]
					output_sequence = X[j] + [1]
				else:
					input_sequence  = [0] + X[j][:-1]
					output_sequence = X[j]

				if debug:
					ops = [train_op, self.cost, self.predict_op]
					_, c, p = sess.run(ops, feed_dict={self.tfX: input_sequence, self.tfY: output_sequence})
					cost += c
					n_total += len(output_sequence)
					n_correct += np.sum(np.array(output_sequence) == p)
				else:
					sess.run(train_op, feed_dict={self.tfX: input_sequence, self.tfY: output_sequence})

			if debug:
				costs.append(cost)
				score = float(n_correct) / n_total
				print('epoch: %d, cost: %.6f, score: %.6f%%' % (i, cost, score*100))

		if debug:
			plt.plot(costs)
			plt.title('Cross-Entropy Cost')
			plt.show()

	def predict(self, prev_words):
		# don't use argmax, so that we can sample
		# from this probability distribution
		return self.session.run(
			self.output_probs,
			feed_dict={self.tfX: prev_words}
		)

	def save(self, filename):
		actual_params = self.session.run(self.params)
		actual_params = [p for p in actual_params]
		np.savez(filename, *actual_params)

	@staticmethod
	def load(filename, session, activation=tf.nn.relu):
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
		rnn = SimpleRNN(D, M, V, session, activation)
		rnn.build(We, Wx, Wh, bh, h0, Wo, bo)
		return rnn

	def generate(self, word2idx, lines=4, print_out=True):
		# convert word2idx -> idx2word
		idx2word = {v:k for k, v in word2idx.items()}
		V = len(word2idx)

		n_lines = 0
		sentences = []

		X = [0]
		sentence = []

		while n_lines < lines:
			PY_X = self.predict(X) # PY_X: (T, V)
			PY_X = PY_X[-1] # (V, )
			P = np.random.choice(V, p=PY_X)
			X.append(P)
			if P > 1:
				word = idx2word[P]
				sentence.append(word)
			elif P == 1:
				n_lines += 1
				sentence.append('\n')
				sentences.append(' '.join(sentence))
				if n_lines < lines:
					X = [0]
					sentence = []

		if print_out:
			for sentence in sentences:
				print(sentence)

		return sentences


def train_poetry(session, D, M, savefile):
	sentences, word2idx = get_robert_frost()
	rnn = SimpleRNN(D, M, len(word2idx), session, activation=tf.nn.relu)
	rnn.fit(sentences, epochs=30, learning_rate=1e-2, debug=True)
	rnn.save(savefile)


def generate_poetry(session, savefile):
	sentences, word2idx = get_robert_frost()
	rnn = SimpleRNN.load(savefile, session, activation=tf.nn.relu)
	rnn.generate(word2idx, lines=4, print_out=True)


if __name__ == '__main__':
	D = 30
	M = 50
	savefile = 'TF_RNN_D30_M50.npz'
	session = tf.InteractiveSession()
	train_poetry(session, D, M, savefile)
	generate_poetry(session, savefile)

