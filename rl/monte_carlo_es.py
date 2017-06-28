# Monte Carlo with Exploring-Starts Method

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# NOTE: this script implements the Monte Carlo Exploring-Starts method
# 		for finding the optimal policy

def play_game(grid, policy):
	# returns a list of states and corresponding returns

	# reset game to start at a random position
	# we need to do this if we have a deterministic policy
	# we would never end up at certains states, but we still want to measure their value
	# this is called 'Exploring Starts' method
	start_states = list(grid.actions.keys())
	start_idx = np.random.choice(len(start_states))
	grid.set_state(start_states[start_idx])

	s = grid.current_state()
	a = np.random.choice(ALL_POSSIBLE_ACTIONS) # first action is uniformly random

	# be aware of timing
	# each triple is s(t), a(t), r(t)
	# but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
	states_actions_rewards = [(s, a, 0)]
	seen_states = set()
	while True:
		old_s = grid.current_state()
		r = grid.move(a)
		s = grid.current_state()

		if s == old_s or s in seen_states:
			# hack so that we don't end up in an infinitely long episode
			# bumping into the wall repeatedly
			states_actions_rewards.append((s, None, -10)) # -10, -20, -50 or -100 is OK
			break
		elif grid.game_over():
			states_actions_rewards.append((s, None, r))
			break
		else:
			a = policy[s]
			states_actions_rewards.append((s, a, r))
		seen_states.add(s)

	# calculate the returns by working backwards from the terminal state
	G = 0
	states_actions_returns = []
	first = True
	for s, a, r in reversed(states_actions_rewards):
		# the value of the terminal state is 0 by definition
		# we should ignore the first state we encounter
		# and ignore the last G, which is meaningless since it doesn't correspond to any move
		if first:
			first = False
		else:
			states_actions_returns.append((s, a, G))
		G = r + GAMMA * G
	states_actions_returns.reverse() # we want it to be in order of state visited
	return states_actions_returns

def max_dict(d):
	# returns the argmax (key) and max (value) from a dictionary
	# put this into a function since we are using it so often
	max_key = None
	max_val = float('-inf')
	for k, v in d.items():
		if v > max_val:
			max_key = k
			max_val = v
	return max_key, max_val


if __name__ == '__main__':
	# use the standard grid again (0 for every step) so that we can compare
	# to iterative policy evaluation
	# grid = standard_grid()
	# try the negative grid too, to see if agent will learn to go past the 'bad spot'
	# in order to minimize number of steps
	grid = negative_grid(step_cost=-0.1)

	# print rewards
	print('Rewards:')
	print_values(grid.rewards, grid)

	# state -> action
	# intialize a random policy
	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

	# print initial policy:
	print('Initial Policy:')
	print_policy(policy, grid)

	# initialize Q(s, a) and returns
	Q = {}
	returns = {} # dictionary of (state, action) -> list of returns we're received
	states = grid.all_states()
	for s in states:
		if s in grid.actions:
			# not a terminal state
			Q[s] = {}
			for a in ALL_POSSIBLE_ACTIONS:
				Q[s][a] = 0 # need to be initialized to something so we can argmax it
				returns[(s, a)] = []
		else:
			# terminal state or state we can't otherwise get to
			pass

	# repeat until convergence
	deltas = []
	for episode in range(5000):
		if episode % 100 == 0:
			print('episode %d finished...' % episode)

		# Policy Evaluation step
		# generate an episode using pi
		biggest_change = 0
		states_actions_returns = play_game(grid, policy)
		seen_state_action_pairs = set()
		for s, a, G in states_actions_returns:
			# check if we have already seen (s, a)
			# called 'First-Visit' MC policy evaluation
			sa = (s, a)
			if sa not in seen_state_action_pairs:
				old_q = Q[s][a]
				returns[sa].append(G)
				Q[s][a] = np.mean(returns[sa])
				biggest_change = max(biggest_change, np.abs(Q[s][a] - old_q))
				seen_state_action_pairs.add(sa)
		deltas.append(biggest_change)

		# Policy Improvement step
		for s in policy.keys():
			policy[s] = max_dict(Q[s])[0]

	plt.plot(deltas)
	plt.show()

	print('Optimal Policy:')
	print_policy(policy, grid)

	# find optimal value function
	V = {}
	for s in policy.keys():
		V[s] = max_dict(Q[s])[1]

	print('Optimal Value Function V(s):')
	print_values(V, grid)

