# Use Q-Learning with RBF Neural Network to solve CartPole
from __future__ import print_function, division
from builtins import range
# NOTE: may need to update the version of future
# pip3 install -U future

# Works best with multiple RBF kernels at var = 0.05, 0.1, 0.5, 1.0

import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from q_learning_bins import plot_running_avg


class SGDRegressor:
	def __init__(self, D, learning_rate=0.1):
		self.w = np.random.randn(D) / np.sqrt(D)
		self.lr = learning_rate

	def partial_fit(self, X, Y):
		# self.w += self.lr * (Y - X.dot(self.w)) * np.squeeze(X)
		self.w += self.lr * (Y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)


class FeatureTransformer:
	def __init__(self, env, n_components=1000, n_samples=20000):
		# observation_examples = np.array([env.observation_space.sample() for i in range(n_samples)])
		# NOTE: state samples are poor, because we get velocity -> infinity
		observation_examples = np.random.random((n_samples, env.observation_space.shape[0]))*2 - 1
		scaler = StandardScaler()
		scaler.fit(observation_examples)

		# Used to converte a state to a featurized representation.
		# We use RBF kernels with different variances to cover different parts of the space
		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=0.05, n_components=n_components)),
			('rbf2', RBFSampler(gamma=1.0 , n_components=n_components)),
			('rbf3', RBFSampler(gamma=0.5 , n_components=n_components)),
			('rbf4', RBFSampler(gamma=0.1 , n_components=n_components))
		])
		feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

		self.dimensions = feature_examples.shape[1]
		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scaler.transform(observations)
		return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
	def __init__(self, env, feature_transformer, learning_rate=0.1):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			model = SGDRegressor(feature_transformer.dimensions, learning_rate)
			self.models.append(model)

	def predict(self, s):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		return np.array([m.predict(X)[0] for m in self.models])

	def update(self, s, a, G):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		self.models[a].partial_fit(X, [G])

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(s))


def play_one(model, eps, gamma, max_iters=2000):
	observation = model.env.reset()
	done = False
	totalreward = 0
	iters = 0
	while not done and iters < max_iters:
		# if we reach 2000, just quit, don't want this going forever
		# the 200 limit seems a bit early
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		if done:
			reward = -200

		# update the model
		next_q = model.predict(observation)
		assert(len(next_q.shape) == 1)
		G = reward + gamma * np.max(next_q)
		model.update(prev_observation, action, G)

		if reward == 1:
			totalreward += reward
		iters += 1

	return totalreward


def main():
	env = gym.make('CartPole-v0')
	ft = FeatureTransformer(env)

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	model = Model(env, ft)
	gamma = 0.99

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1.0 / np.sqrt(n + 1)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		if n % 100 == 0:
			print('episode:', n, 'total reward:', totalreward, 'eps:', eps, 'avg reward (last 100):', totalrewards[max(0, n-100):(n+1)].mean())

	print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
	print('total steps:', totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)


if __name__ == '__main__':
	main()

