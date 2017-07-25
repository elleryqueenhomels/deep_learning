# Use Q-Learning with TD(lambda) to solve MountainCar

import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer, plot_cost_to_go, plot_running_avg


class BaseModel:
	def __init__(self, D):
		self.w = np.random.randn(D) / np.sqrt(D)

	# theta <- theta + alpha * delta * eligibility
	# delta = target - prediction
	# target = R(t+1) + gamma * V(S(t+1))
	# prediction = V(S(t))
	# eligibility: e(0) = 0, e(t) = gradient + gamma * lambda * e(t-1)
	# gradient = d[V(S(t))] / d[theta]
	def partial_fit(self, X, target, eligibility, lr=1e-2):
		prediction = X.dot(self.w)
		self.w += lr * (target - prediction) * eligibility

	def predict(self, X):
		X = np.array(X)
		return X.dot(self.w)


# Holds one BaseModel for each action
class Model:
	def __init__(self, env, feature_transformer):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer

		D = feature_transformer.dimensions
		# eligibility: e(0) = 0, e(t) = gradient + gamma * lambda * e(t-1)
		# gradient = d[V(S(t))] / d[theta]
		self.eligibilities = np.zeros((env.action_space.n, D))
		for i in range(env.action_space.n):
			model = BaseModel(D)
			self.models.append(model)

	def predict(self, s):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		assert(len(X.shape) == 2)
		return np.array([model.predict(X)[0] for model in self.models])

	def update(self, s, a, G, gamma, lambda_):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		assert(len(X.shape) == 2)
		# self.eligibilities[a] *= gamma * lambda_
		self.eligibilities *= gamma * lambda_
		# grad of V(S(t)) wrt. theta is X, X.shape is (1, D), X[0].shape is (D, )
		self.eligibilities[a] += X[0]
		self.models[a].partial_fit(X[0], G, self.eligibilities[a])

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(s))


def play_one(model, eps, gamma, lambda_, max_iters=10000):
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
		model.update(prev_observation, action, G, gamma, lambda_)

		totalreward += reward
		iters += 1

	return totalreward


def main():
	gym.envs.register(
		id='MyMountainCar-v0',
		entry_point='gym.envs.classic_control:MountainCarEnv',
		max_episode_steps=10000,
		reward_threshold=-5500,
	)
	env = gym.make('MyMountainCar-v0')
	# env = gym.make('MountainCar-v0')
	ft = FeatureTransformer(env, n_components=1000, n_samples=20000)

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	model = Model(env, ft)
	gamma = 0.99
	lambda_ = 0.7

	N = 300
	totalrewards = np.empty(N)
	for n in range(N):
		# eps = 1.0 / (0.1 * n + 1)
		eps = 0.5 / np.sqrt(n + 1)
		# eps = 0.1 * (0.97**n)
		totalreward = play_one(model, eps, gamma, lambda_)
		totalrewards[n] = totalreward
		if n % 20 == 0:
			print('episode: %d, current reward: %s, last 100 episodes avg reward: %s' % (n, totalreward, totalrewards[max(0, n-99):(n+1)].mean()))

	print('avg reward for last 100 episodes: %s' % totalrewards[-100:].mean())
	print('total steps: %s' % -totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)

	# plot the negative optimal state-value function
	plot_cost_to_go(env, model)


if __name__ == '__main__':
	main()

