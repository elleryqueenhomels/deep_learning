# Use Q-Learning with N-Step to solve MountainCar

import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime

import q_learning
from q_learning import FeatureTransformer, Model, plot_cost_to_go, plot_running_avg


class SGDRegressor:
	def __init__(self, learning_rate=1e-2, **kwargs):
		self.w = None
		self.lr = learning_rate

	def partial_fit(self, X, Y):
		if self.w is None:
			D = X.shape[1]
			self.w = np.random.randn(D) / np.sqrt(D)
		self.w += self.lr * (Y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)

# replace sklearn SGDRegressor
q_learning.SGDRegressor = SGDRegressor

# calculate everything up to max[Q(s, a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
# def calculate_return_before_prediction(rewards, gamma):
# 	res = 0
# 	for r in reversed(rewards[1:]):
# 		res += r + gamma*res
# 	res += rewards[0]
# 	return res

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, n=5, max_iters=10000):
	observation = model.env.reset()
	done = False
	totalreward = 0
	rewards = []
	states = []
	actions = []
	iters = 0
	# array of [gamma^0, gamma^1, gamma^2, ... , gamma^(n-1)]
	multiplier = np.array([gamma]*n)**np.arange(n)

	while not done and iters < max_iters:
		action = model.sample_action(observation, eps)

		states.append(observation)
		actions.append(action)

		# prev_observation = observation
		observation, reward, done, info = model.env.step(action)

		rewards.append(reward)

		# update the model
		if len(rewards) >= n:
			# return_up_to_prediction = calculate_return_before_prediction(rewards, gamma)
			return_up_to_prediction = multiplier.dot(rewards[-n:])
			G = return_up_to_prediction + (gamma**n) * np.max(model.predict(observation))
			model.update(states[-n], actions[-n], G)

		totalreward += reward
		iters += 1

	# empty the cache
	if n == 1:
		rewards = []
		states = []
		actions = []
	else:
		rewards = rewards[-n+1:]
		states = states[-n+1:]
		actions = actions[-n+1:]

	# new version of gym cuts us off at 200 steps
	# even if we haven't reached the goal.
	# it's not good to do this UNLESS we have reached the goal.
	# we are 'really done' if position >= 0.5
	if observation[0] >= 0.5:
		# we actually made it to the goal
		# so all the future rewards are 0
		while len(rewards) > 0:
			G = multiplier[:len(rewards)].dot(rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
	else:
		# we did not make it to the goal
		# for MountainCar, every step reward is -1
		while len(rewards) > 0:
			guess_rewards = rewards + [-1] * (n - len(rewards))
			G = multiplier.dot(guess_rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)

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

	model = Model(env, ft, learning_rate=1e-2)
	gamma = 0.99

	N = 300
	totalrewards = np.empty(N)
	for n in range(N):
		# eps = 1.0 / (0.1*n + 1)
		eps = 1.0 / np.sqrt(n + 1)
		# eps = 0.1 * (0.97**n)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		print('episode: %d, current reward: %s' % (n, totalreward))

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

