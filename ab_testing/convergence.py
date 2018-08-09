# From the course: Bayesian Machine Learning in Python: A/B Testing
# A demo to illustrating how Bayesian Sampling converges to the best CTR (the best bandit).

import numpy as np
import matplotlib.pyplot as plt
from bayesian_bandit import Bandit


def run_experiment(probabilities, N):
	bandits = [Bandit(p) for p in probabilities]

	data = np.empty(N)

	for i in range(N):
		# Thompson Sampling
		j = np.argmax([b.sample() for b in bandits])
		x = bandits[j].pull()
		bandits[j].update(x)

		# for the plot
		data[i] = x

	cumulative_average_ctr = np.cumsum(data) / (np.arange(N) + 1)

	# plot moving average ctr
	plt.plot(cumulative_average_ctr)
	for p in probabilities:
		plt.plot(np.ones(N)*p, label='real p: %.3f' % p)
	plt.legend()
	plt.ylim((0,1))
	plt.xscale('log')
	plt.show()


if __name__ == '__main__':
	run_experiment([0.2, 0.25, 0.3], 100000)
