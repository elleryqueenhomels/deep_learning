# Use Q-Learning with RBF Neural Network to solve MountainCar
from __future__ import print_function, division
from builtins import range
# NOTE: may need to update the version of future
# pip3 install -U future
#
# NOTE: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means the agent can't learn as much in the earlier episodes
# since they are no longer as long.

import os
import sys
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta=0.01, power_t=0.25, warm_start=False, average=False


class FeatureTransformer:
	def __init__(self, env, n_components=500, n_samples=10000):
		observation_examples = np.array([env.observation_space.sample() for i in range(n_samples)])
		scaler = StandardScaler()
		scaler.fit(observation_examples)

		# Used to converte a state to a featurized representation.
		# We use RBF kernels with different variances to cover different parts of the space
		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
			('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
			('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
			('rbf4', RBFSampler(gamma=0.5, n_components=n_components))
		])
		example_features = featurizer.fit_transform(scaler.transform(observation_examples))

		self.dimensions = example_features.shape[1]
		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scaler.transform(observations)
		# assert(len(scaled.shape) == 2)
		return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
	def __init__(self, env, feature_transformer, learning_rate='constant'):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			model = SGDRegressor(learning_rate=learning_rate)
			model.partial_fit(feature_transformer.transform([env.reset()]), [0])
			self.models.append(model)

	def predict(self, s):
		X = self.feature_transformer.transform([s])
		assert(len(X.shape) == 2)
		return np.array([m.predict(X)[0] for m in self.models])

	def update(self, s, a, G):
		X = self.feature_transformer.transform([s])
		assert(len(X.shape) == 2)
		self.models[a].partial_fit(X, [G])

	def sample_action(self, s, eps):
		# eps = 0
		# Technically, we don't need to do Epsilon-Greedy
		# because SGDRegressor predicts 0 for all states
		# until they are updated. This works as the 
		# 'Optimistic Initial Values' method, since all
		# the rewards for Mountain Car are -1.
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, max_iters=10000):
	observation = model.env.reset()
	done = False
	totalreward = 0
	iters = 0
	while not done and iters < max_iters:
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		# update the model
		G = reward + gamma * np.max(model.predict(observation))
		model.update(prev_observation, action, G)

		totalreward += reward
		iters += 1

	return totalreward


def plot_cost_to_go(env, estimator, num_tiles=20):
	x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
	y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
	X, Y = np.meshgrid(x, y)
	# both X and Y will be of shape (num_tiles, num_tiles)
	Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
	# Z will be of shape (num_tiles, num_tiles)

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Cost-To-Go == -V(s)')
	ax.set_title('Cost-To-Go Function')
	fig.colorbar(surf)
	plt.show()


def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for n in range(N):
		running_avg[n] = totalrewards[max(0, n-100):(n+1)].mean()
	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()


def main():
	gym.envs.register(
		id='MyMountainCar-v0',
		entry_point='gym.envs.classic_control:MountainCarEnv',
		max_episode_steps=10000,
		reward_threshold=-5500,
	)
	env = gym.make('MyMountainCar-v0')
	# env = gym.make('MountainCar-v0')
	ft = FeatureTransformer(env)

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	model = Model(env, ft, 'constant')
	# learning_rate == 1e-4
	# eps = 1.0
	gamma = 0.99

	N = 300
	totalrewards = np.empty(N)
	for n in range(N):
		# eps = 1.0 / (0.1 * n + 1)
		eps = 0.1 * (0.97**n)
		# eps = 0.5 / np.sqrt(n + 1)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		print('episode:', n, 'total reward:', totalreward)

	print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
	print('total steps:', -totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)

	# plot the negative optimal state-value function
	plot_cost_to_go(env, model)


if __name__ == '__main__':
	main()

