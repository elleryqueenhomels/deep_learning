# Use Random Search to find params for CartPole
from __future__ import print_function, division
from builtins import range
# NOTE: may need to update the version of future
# pip3 install -U future

import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
	return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params, max_steps=10000):
	observation = env.reset()
	done = False
	t = 0

	while not done and t < max_steps:
		# env.render()
		t += 1
		action = get_action(observation, params)
		observation, reward, done, info = env.step(action)
		if done:
			break

	return t


def play_multiple_episodes(env, params, num_episodes, max_steps=10000):
	episode_lengths = 0

	for i in range(num_episodes):
		episode_lengths += play_one_episode(env, params, max_steps=max_steps)

	avg_length = episode_lengths / num_episodes
	print('avg length:', avg_length)
	return avg_length


def random_search(env, params_dimens, times=100, num_episodes=100):
	episode_lengths = []
	best = 0
	params = None
	for t in range(times):
		new_params = np.random.random(params_dimens)*2 - 1
		avg_length = play_multiple_episodes(env, new_params, num_episodes)
		episode_lengths.append(avg_length)

		if avg_length > best:
			best = avg_length
			params = new_params

	return episode_lengths, params


if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	episode_lengths, params = random_search(env, 4)

	plt.plot(episode_lengths)
	plt.show()

	# play a final set of episodes
	print('*** Final run with final weights ***')
	play_multiple_episodes(env, params, 100)

