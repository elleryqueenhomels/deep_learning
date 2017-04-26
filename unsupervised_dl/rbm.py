import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import init_weights, error_rate, get_mnist
from autoencoder import DNN


class RBM(object):
	def __init__(self, M, an_id):
		self.M = M
		self.id = an_id
		self.rng = RandomStreams()

	def fit(self, X, epochs=1, batch_sz=100, learning_rate=0.5, momentum=0.99, show_fig=False):
		# Use float32 for GPU accelerated
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		one = np.float32(1)

		X = X.astype(np.float32)
		N, D = X.shape

		if batch_sz <= 0 or batch_sz >= N:
			batch_sz = N // 100
		n_batches = N // batch_sz

		W = init_weights((D, self.M))
		c = np.zeros(self.M, dtype=np.float32)
		b = np.zeros(D, dtype=np.float32)
		self.W = theano.shared(W, 'W_%s' % self.id)
		self.c = theano.shared(c, 'c_%s' % self.id)
		self.b = theano.shared(b, 'b_%s' % self.id)
		self.params = [self.W, self.c, self.b]
		self.forward_params = [self.W, self.c]

		dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

		X_in = T.fmatrix('X_%s' % self.id)

		# attach it to the object so it can be used later
		# must be sigmoidial because the output is also a sigmoid
		FH = self.forward_hidden(X_in)
		self.forward_hidden_op = theano.function(inputs=[X_in], outputs=FH)

		# we don't use this cost to do any updates
		# but we would like to see how this cost function changes
		# as we do Contrastive Divergence
		X_hat = self.forward_output(X_in)
		cost = -T.mean(X_in * T.log(X_hat) + (one - X_in) * T.log(one - X_hat))
		cost_op = theano.function(inputs=[X_in], outputs=cost)

		# do one round of Gibbs Sampling to obtain X_sample
		H = self.sample_h_given_v(X_in)
		X_sample = self.sample_v_given_h(H)

		# define the objective, updates, and train function
		objective = T.mean(self.free_energy(X_in)) - T.mean(self.free_energy(X_sample))

		# need to consider X_sample as constant because Theano can't take the gradient of random numbers
		updates = [(p, p - lr*T.grad(objective, p, consider_constant=[X_sample])) for p in self.params]

		train_op = theano.function(inputs=[X_in], updates=updates)

		print('\nTraining Restricted Boltzmann Machine %s:' % self.id)
		costs = []
		for i in range(epochs):
			X = shuffle(X)
			for j in range(n_batches):
				Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
				train_op(Xbatch)

				if j % 20 == 0:
					the_cost = cost_op(X) # technically we could also get the cost for Xtest here
					costs.append(the_cost)
					print('epoch=%d, batch=%d, n_batches=%d: cost=%.6f' % (i, j, n_batches, the_cost))

		if show_fig:
			plt.plot(costs)
			plt.title('Costs in Restricted Boltzmann Machine %s' % self.id)
			plt.show()

	def free_energy(self, V):
		one = np.float32(1)
		return -V.dot(self.b) - T.sum(T.log(one + T.exp(V.dot(self.W) + self.c)), axis=1)

	def sample_h_given_v(self, V):
		p_h_given_v = T.nnet.sigmoid(V.dot(self.W) + self.c)
		h_sample = self.rng.binomial(n=1, p=p_h_given_v, size=p_h_given_v.shape, dtype='float32')
		return h_sample

	def sample_v_given_h(self, H):
		p_v_given_h = T.nnet.sigmoid(H.dot(self.W.T) + self.b)
		v_sample = self.rng.binomial(n=1, p=p_v_given_h, size=p_v_given_h.shape, dtype='float32')
		return v_sample

	def forward_hidden(self, X):
		return T.nnet.sigmoid(X.dot(self.W) + self.c)

	def forward_output(self, X):
		Z = self.forward_hidden(X)
		return T.nnet.sigmoid(Z.dot(self.W.T) + self.b)

	@staticmethod
	def createFromArray(W, c, b, an_id):
		rbm = RBM(W.shape[1], an_id)
		rbm.W = theano.shared(W.astype(np.float32), 'W_%s' % rbm.id)
		rbm.c = theano.shared(c.astype(np.float32), 'c_%s' % rbm.id)
		rbm.b = theano.shared(b.astype(np.float32), 'b_%s' % rbm.id)
		rbm.params = [rbm.W, rbm.c, rbm.b]
		rbm.forward_params = [rbm.W, rbm.c]
		return rbm


def main():
	Xtrain, Ytrain, Xtest, Ytest = get_mnist(21000)
	dnn = DNN([1000, 750, 500], UnsupervisedModel=RBM)
	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)

	# we compare with no pretraining in autoencoder.py


if __name__ == '__main__':
	main()

