# Use Q-Learning with Bins to solve CartPole
from __future__ import print_function, division
from builtins import range
# NOTE: may need to update the version of future
# 		if builtins is not defined
# pip3 install -U future

import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime


# turns list of integers into an int
# Ex.
# build_state([1,2,3,4,5]) -> 12345
def build_state(features):
	return int(''.join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
	return np.digitize(x=[value], bins=bins)[0]


class FeatureTransformer:
	def __init__(self):
		# NOTE: to make this better we could look at how often each bin was
		# actually used while running the script.
		# It's not clear from the high/low values nor sample() what values
		# we really expect to get.
		self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
		self.cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf) (these might not be good values)
		self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
		self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf) (these might not be good values)

	def transform(self, observation):
		# returns an int
		cart_pos, cart_vel, pole_ang, pole_vel = observation
		return build_state([
			to_bin(cart_pos, self.cart_position_bins),
			to_bin(cart_vel, self.cart_velocity_bins),
			to_bin(pole_ang, self.pole_angle_bins),
			to_bin(pole_vel, self.pole_velocity_bins),
		])


class Model:
	def __init__(self, env, feature_transformer):
		self.env = env
		self.feature_transformer = feature_transformer

		num_states = 10**env.observation_space.shape[0]
		num_actions = env.action_space.n
		self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

	def predict(self, s):
		x = self.feature_transformer.transform(s)
		return self.Q[x]

	def update(self, s, a, G, alpha=1e-2):
		x = self.feature_transformer.transform(s)
		self.Q[x, a] += alpha*(G - self.Q[x, a])

	def sample_action(self, s, eps):
		# Epsilon-Greedy
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			p = self.predict(s)
			return np.argmax(p)


def play_one(model, eps, gamma, max_iters=10000):
	observation = model.env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < max_iters:
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		totalreward += reward

		if done and iters < 199:
			reward = -300

		# update the model
		G = reward + gamma * np.max(model.predict(observation))
		model.update(prev_observation, action, G)

		iters += 1

	return totalreward


def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for n in range(N):
		running_avg[n] = totalrewards[max(0, n-99):(n+1)].mean()
	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()


def main():
	env = gym.make('CartPole-v0')
	ft = FeatureTransformer()

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	model = Model(env, ft)
	gamma = 0.9

	N = 10000
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1.0 / np.sqrt(n + 1)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		if n % 100 == 0:
			print('episodes:', n, 'total reward:', totalreward, 'eps:', eps)

	print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
	print('total steps:', totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)


if __name__ == '__main__':
	main()

