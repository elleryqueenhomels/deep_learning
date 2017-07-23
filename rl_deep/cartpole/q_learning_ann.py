# Use Q-Learning with ANN to solve CartPole

import os
import sys
import gym
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


def init_weight_and_bias(Mi, Mo):
	Mi, Mo = int(Mi), int(Mo)
	W = np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)
	b = np.zeros(Mo)
	return W, b


class HiddenLayer:
	def __init__(self, Mi, Mo, activation):
		W, b = init_weight_and_bias(Mi, Mo)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.params = [self.W, self.b]
		self.f = activation

	def forward(self, X):
		if self.f is not None:
			return self.f(X.dot(self.W) + self.b)
		else:
			return X.dot(self.W) + self.b


class ANNRegressor:
	def __init__(self, D, hidden_layer_sizes, activation=None, learning_rate=0.1):
		self.hidden_layers = []
		Mi = D
		for Mo in hidden_layer_sizes:
			h = HiddenLayer(Mi, Mo, activation)
			self.hidden_layers.append(h)
			Mi = Mo
		W, b = init_weight_and_bias(Mi, 1)
		self.W = theano.shared(W)
		self.b = theano.shared(b)

		# collect params for later use
		self.params = [self.W, self.b]
		for h in reversed(self.hidden_layers):
			self.params += h.params

		# set up theano functions and variables
		thX = T.matrix('X')
		thY = T.vector('Y')

		Y_hat = self.forward(thX).flatten()
		delta = thY - Y_hat
		cost = delta.dot(delta)
		grads = T.grad(cost, self.params)
		updates = [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

		self.train_op = theano.function(
			inputs=[thX, thY],
			updates=updates,
		)

		self.predict_op = theano.function(
			inputs=[thX],
			outputs=Y_hat,
		)

	def forward(self, X):
		Z = X
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return Z.dot(self.W) + self.b

	def partial_fit(self, X, Y):
		self.train_op(X, Y)

	def predict(self, X):
		return self.predict_op(X)


class FeatureTransformer:
	def __init__(self, env, n_components=500, n_samples=20000):
		# observation_examples = np.array([env.observation_space.sample() for i in range(n_samples)])
		# NOTE: state samples are poor in CartPole while using uniformly sampling, because we get velocity -> infinity
		D = env.observation_space.shape[0]
		observation_examples = np.random.random((n_samples, D))*2 - 1
		scaler = StandardScaler()
		scaler.fit(observation_examples)

		# Used to converte a state to a featurized representation
		# We use RBF kernels with different variances to cover different parts of the space
		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=0.05, n_components=n_components)),
			('rbf2', RBFSampler(gamma=1.0 , n_components=n_components)),
			('rbf3', RBFSampler(gamma=0.5 , n_components=n_components)),
			('rbf4', RBFSampler(gamma=0.1 , n_components=n_components)),
		])
		feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

		self.dimensions = feature_examples.shape[1]
		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scaler.transform(observations)
		return self.featurizer.transform(scaled)


# Holds one ANNRegressor for each action
class Model:
	def __init__(self, env, ann_layers, feature_transformer=None, activation=None, learning_rate=0.1):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		if feature_transformer is None:
			D = env.observation_space.shape[0]
		else:
			D = feature_transformer.dimensions
		for i in range(env.action_space.n):
			model = ANNRegressor(D, ann_layers, activation=activation, learning_rate=learning_rate)
			self.models.append(model)

	def predict(self, s):
		X = np.atleast_2d(s)
		if self.feature_transformer is not None:
			X = self.feature_transformer.transform(X)
		return np.array([model.predict(X)[0] for model in self.models])

	def update(self, s, a, G):
		X = np.atleast_2d(s)
		if self.feature_transformer is not None:
			X = self.feature_transformer.transform(X)
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
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		if done:
			reward = -200

		# update the model
		next_q = model.predict(observation)
		assert(len(next_q.shape) == 1)
		assert(next_q.shape[0] == model.env.action_space.n)
		G = reward + gamma * np.max(next_q)
		model.update(prev_observation, action, G)

		if reward == 1:
			totalreward += reward

		iters += 1

	return totalreward


def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for n in range(N):
		running_avg[n] = totalrewards[max(0, n-99):(n+1)].mean()
	plt.plot(running_avg)
	plt.title('Running Average (last 100 episodes)')
	plt.show()


def main():
	gym.envs.register(
		id='MyCartPole-v0',
		entry_point='gym.envs.classic_control:CartPoleEnv',
		max_episode_steps=10000,
		reward_threshold=9975.0,
	)
	env = gym.make('MyCartPole-v0')
	ft = FeatureTransformer(env, n_components=500, n_samples=20000)

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	model = Model(env, ann_layers=[500, 500], feature_transformer=ft, activation=T.tanh, learning_rate=1e-1)
	gamma = 0.99

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1.0 / np.sqrt(n + 1)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		if n % 100 == 0:
			print('episode: %d, reward: %s, eps: %s, avg reward(last 100): %s' % (n, totalreward, eps, totalrewards[max(0, n-99):(n+1)].mean()))

	print('\navg reward for last 100 episodes: %s' % totalrewards[-100:].mean())
	print('total steps: %s\n' % totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)


if __name__ == '__main__':
	main()

