# Epsilon-Greedy Algorithm

import numpy as np
import matplotlib.pyplot as plt


class Bandit(object):
	def __init__(self, m):
		self.m = m # true mean
		self.mean = 0
		self.N = 0

	def pull(self):
		return np.random.randn() + self.m

	def update(self, x):
		self.N += 1
		self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(means, eps, N):
	bandits = [Bandit(mean) for mean in means]
	n_bandits = len(bandits)

	data = np.empty(N)

	for i in range(N):
		# Epsilon-Greedy
		p = np.random.random()
		if p < eps:
			# explore
			j = np.random.choice(n_bandits)
		else:
			# exploit
			j = np.argmax([b.mean for b in bandits])
		x = bandits[j].pull()
		bandits[j].update(x)

		# for the plot
		data[i] = x

	cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

	# plot moving average ctr
	plt.plot(cumulative_average)
	for mean in means:
		plt.plot(np.ones(N) * mean)
	plt.xscale('log')
	plt.title('Epsilon-Greedy')
	plt.show()

	for i, b in enumerate(bandits):
		print('Bandit %d mean: %f' % (i, b.mean))

	return cumulative_average


if __name__ == '__main__':
	c_1  = run_experiment([1.0, 2.0, 3.0], 0.1,  100000)
	c_05 = run_experiment([1.0, 2.0, 3.0], 0.05, 100000)
	c_01 = run_experiment([1.0, 2.0, 3.0], 0.01, 100000)

	# log scale plot
	plt.plot(c_1,  label='eps = 0.1')
	plt.plot(c_05, label='eps = 0.05')
	plt.plot(c_01, label='eps = 0.01')
	plt.legend()
	plt.xscale('log')
	plt.title('Epsilon-Greedy')
	plt.show()

	# linear plot
	plt.plot(c_1,  label='eps = 0.1')
	plt.plot(c_05, label='eps = 0.05')
	plt.plot(c_01, label='eps = 0.01')
	plt.legend()
	plt.title('Epsilon-Greedy')
	plt.show()

