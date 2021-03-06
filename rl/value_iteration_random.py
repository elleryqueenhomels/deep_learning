# Value Iteration in Windy Gridworld

import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# next state and reward will now have some randomness
# you'll go in your desired direction with probability 0.5
# you'll go in a random direction a' != a with probability 0.5/3

def main():
	# this grid gives you a reward of -0.1 for every non-terminal state
	# we want to see if this will encourage finding a shorter path to the goal
	grid = negative_grid(step_cost=-1.0)
	# grid = negative_grid(step_cost=-0.1)
	# grid = standard_grid()

	# print rewards
	print('Rewards:')
	print_values(grid.rewards, grid)

	# state -> action
	# we'll randomly choose an action and update as we learn
	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

	# print initial policy
	print('Intial Policy:')
	print_policy(policy, grid)

	# initialize V(s)
	V = {}
	states = grid.all_states()
	for s in states:
		if s in grid.actions:
			V[s] = np.random.random()
		else:
			# terminal state
			V[s] = 0

	# repeat until convergence
	# V(s) = max[a]{ sum[s',r]{ p(s',r|s,a) * [r + gamma * V(s')] } }
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]

			# V(s) only has value if it is not a terminal state
			if s in policy:
				new_v = float('-inf')
				for a in ALL_POSSIBLE_ACTIONS:
					v = 0
					for a2 in ALL_POSSIBLE_ACTIONS:
						if a2 == a:
							p = 0.5
						else:
							p = 0.5 / 3
						grid.set_state(s)
						r = grid.move(a2)
						v += p * (r + GAMMA * V[grid.current_state()])
					if v > new_v:
						new_v = v
				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(V[s] - old_v))

		if biggest_change < SMALL_ENOUGH:
			break

	# find a policy that leads to optimal value function
	for s in policy.keys():
		best_a = None
		best_value = float('-inf')
		# loop through all possible actions to find the best current action
		for a in ALL_POSSIBLE_ACTIONS:
			v = 0
			for a2 in ALL_POSSIBLE_ACTIONS:
				if a2 == a:
					p = 0.5
				else:
					p = 0.5 / 3
				grid.set_state(s)
				r = grid.move(a2)
				v += p * (r + GAMMA * V[grid.current_state()])
			if v > best_value:
				best_value = v
				best_a = a
		policy[s] = best_a

	# our goal here is to verify that we get the same answer as with Policy Iteration
	print('Optimal Value Functions:')
	print_values(V, grid)
	print('Optimal Policy')
	print_policy(policy, grid)
	# if step_cost = -1.0, results:
	# every move is as bad as losing, so AI will end the game as quickly as possible


if __name__ == '__main__':
	main()

