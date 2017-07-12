# Apply Approximation Method to TD(0) using semi-gradient for Prediction Problem

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import play_game, SMALL_ENOUGH, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS

# NOTE: this is only policy evaluation, not optimization

class Model:
	def __init__(self):
		self.theta = np.random.randn(4) / 2

	def s2x(self, s):
		return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

	def predict(self, s):
		x = self.s2x(s)
		return self.theta.dot(x)

	def grad(self, s):
		return self.s2x(s)


if __name__ == '__main__':
	# use the standard grid again (0 for every step) so that we can compare
	# to iterative policy evaluation
	grid = standard_grid()

	# print rewards
	print('Rewards:')
	print_values(grid.rewards, grid)

	# state -> action
	policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
	}

	model = Model()
	deltas = []

	# repeat until convergence
	k = 1.0
	for episode in range(20000):
		if episode % 10 == 0:
			k += 1e-2
		alpha = ALPHA / k # adaptive learning rate
		biggest_change = 0

		# generate an episode using pi
		states_and_rewards = play_game(grid, policy)

		for t in range(len(states_and_rewards) - 1):
			s, _ = states_and_rewards[t]
			s2, r = states_and_rewards[t + 1]
			# we will update V(s) AS we experience the episode
			old_theta = model.theta.copy()
			if grid.is_terminal(s2):
				target = r
			else:
				target = r + GAMMA*model.predict(s2)
			model.theta += alpha*(target - model.predict(s))*model.grad(s)
			biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())

		deltas.append(biggest_change)

	plt.plot(deltas)
	plt.show()

	# obtain predicted values
	V = {}
	states = grid.all_states()
	for s in states:
		if s in grid.actions:
			V[s] = model.predict(s)
		else:
			# terminal state or state we can't otherwise get to
			V[s] = 0

	print('Values:')
	print_values(V, grid)
	print('Policy:')
	print_policy(policy, grid)

