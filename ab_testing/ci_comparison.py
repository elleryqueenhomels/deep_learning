# From the course: Bayesian Machine Learning in Python: A/B Testing
# Confidence Interval Approximation VS. Beta Posterior

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

T = 1001 # number of iterations (coin tosses)
true_ctr = 0.5
a, b = 1, 1 # beta priors (At first, a = 1 & b = 1, is Uniform Distribution)
data = np.empty(T)
plot_indices = [10, 20, 30, 50, 100, 200, 500, 750, 1000]

for i in range(T):
	x = 1 if np.random.random() < true_ctr else 0
	data[i] = x

	# update a and b --> Thompson Sampling (Bayesian Method)
	a += x
	b += 1 - x

	if i in plot_indices:
		# Maximum Likelihood Estimate of CTR
		n = i + 1 # number of samples collected so far
		p = data[:n].mean()
		std = np.sqrt(p*(1-p)/n)

		# Gaussian
		x = np.linspace(0, 1, 1000)
		g = norm.pdf(x, loc=p, scale=std)
		plt.plot(x, g, label='Gaussian Approximation')

		# Beta Posterior
		posterior = beta.pdf(x, a=a, b=b)
		plt.plot(x, posterior, label='Beta Posterior')

		plt.title('N = %s, P = %.4f' % (n, p))
		plt.legend()
		plt.show()
