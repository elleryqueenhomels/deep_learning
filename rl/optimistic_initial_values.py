# Optimistic Initial Values Strategy

import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_experiment as run_experiment_eps


class Bandit(object):
	def __init__(self, m):
		self.m = m # true mean
		self.mean = 10
		self.N = 0

	def pull(self):
		return np.random.randn() + self.m

	def update(self, x):
		self.N += 1
		self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(means, N):
	bandits = [Bandit(mean) for mean in means]
	n_bandits = len(bandits)

	data = np.empty(N)

	for i in range(N):
		# Optimistic Initial Values
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
	plt.title('Optimistic Initial Values')
	plt.show()

	for i, b in enumerate(bandits):
		print('Bandit %d mean: %f' % (i, b.mean))

	return cumulative_average


if __name__ == '__main__':
	c_1 = run_experiment_eps([1.0, 2.0, 3.0], 0.1,  100000)
	oiv = run_experiment([1.0, 2.0, 3.0], 100000)

	# log scale plot
	plt.plot(c_1, label='eps = 0.1')
	plt.plot(oiv, label='optimistic')
	plt.legend()
	plt.xscale('log')
	plt.title('Epsilon-Greedy vs. Optimistic Initial Values')
	plt.show()

	# linear plot
	plt.plot(c_1, label='eps = 0.1')
	plt.plot(oiv, label='optimistic')
	plt.legend()
	plt.title('Epsilon-Greedy vs. Optimistic Initial Values')
	plt.show()

