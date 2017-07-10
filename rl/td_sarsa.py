# Temporal Difference using SARSA for Control Problem

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from td0_prediction import random_action

GAMMA = 0.9 # discount factor
ALPHA = 0.1 # initial learning rate
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == '__main__':
	# NOTE: if we use the standard grid, there's a good chance we will end up with
	# suboptimal policies
	# e.g.
	# -------------------------
	# |  R  |  R  |  R  |     |
	# -------------------------
	# |  R* |     |  U  |     |
	# -------------------------
	# |  U  |  R  |  U  |  L  |
	# -------------------------
	# since going R at (1, 0) (shown with a *) incurs no cost, it's OK to keep doing that.
	# we'll either end up staying in the same spot, or back to the start (2, 0), at which
	# point we should then just go back up, or at (0, 0), at which point we can continue
	# on right.
	# instead, let's penalize each movement so the agent will learn to find a shorter route.
	#
	# grid = standard_grid()
	grid = negative_grid(step_cost=-0.1)

	# print rewards
	print('Rewards:')
	print_values(grid.rewards, grid)

	# no policy initialization, we will derive our poliy from most recent Q

	# initialize Q(s, a)
	Q = {}
	states = grid.all_states()
	for s in states:
		Q[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			Q[s][a] = 0

	# let's also keep track of how many times Q[s] has been updated
	update_counts = {} # for debugging
	update_counts_sa = {} # for calculating effective learning rate
	for s in states:
		update_counts_sa[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			update_counts_sa[s][a] = 1.0

	# repeat until convergence
	t = 1.0 # for decaying epsilon-greedy
	deltas = []
	for episode in range(10000):
		if episode % 100 == 0:
			t += 1e-2
		if episode % 2000 == 0:
			print('Episode %d finished...' % episode) # for debugging

		# instead of 'generating' an episode, we will PLAY
		# an episode within this loop
		s = (2, 0) # start state
		grid.set_state(s)

		a = max_dict(Q[s])[0]
		a = random_action(a, eps=0.5/t) # decaying epsilon-greedy
		biggest_change = 0
		while not grid.game_over():
			r = grid.move(a)
			s2 = grid.current_state()

			# we need the next action as well since Q(s, a) depends on Q(s', a')
			# if s2 not in policy then it's a terminal state, all Q are 0
			a2 = max_dict(Q[s2])[0]
			a2 = random_action(a2, eps=0.5/t) # decaying epsilon-greedy

			# we will update Q(s, a) AS we experience the episode
			alpha = ALPHA / update_counts_sa[s][a]
			update_counts_sa[s][a] += 0.005

			old_qsa = Q[s][a]
			Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2] - Q[s][a])
			biggest_change = max(biggest_change, np.abs(Q[s][a] - old_qsa))

			# we would like to know how often Q(s) has been updated too
			update_counts[s] = update_counts.get(s, 0) + 1

			# next state becomes current state
			s = s2
			a = a2

		deltas.append(biggest_change)

	plt.plot(deltas)
	plt.show()

	# determine the optimal policy from Q*
	# find V* from Q*
	policy = {}
	V = {}
	for s in grid.actions.keys():
		a, max_q = max_dict(Q[s])
		policy[s] = a
		V[s] = max_q

	# what's the proportion of time we spend updating each part of Q?
	print('Update Counts:')
	total = np.sum(list(update_counts.values()))
	for k, v in update_counts.items():
		update_counts[k] = float(v) / total
	print_values(update_counts, grid)

	print('Optimal Values:')
	print_values(V, grid)
	print('Optimal Policy:')
	print_policy(policy, grid)

