# Use Policy Gradient to solve CartPole

import os
import sys
import gym
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from q_learning_bins import plot_running_avg


class HiddenLayer:
	def __init__(self, Mi, Mo, activation=T.tanh, use_bias=True):
		self.W = theano.shared(np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo))
		self.params = [self.W]
		self.use_bias = use_bias
		if use_bias:
			self.b = theano.shared(np.zeros(Mo))
			self.params.append(self.b)
		self.f = activation

	def forward(self, X):
		if self.use_bias:
			a = X.dot(self.W) + self.b
		else:
			a = X.dot(self.W)
		return self.f(a)


# approximate pi(a | s)
class PolicyModel:
	def __init__(self, D, K, hidden_layer_sizes, activation=T.tanh, learning_rate=1e-3, momentum=0.7, decay=0.999):
		# create the graph
		# K == number of actions
		lr = learning_rate
		mu = momentum
		decay = decay

		self.layers = []
		Mi = D
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=activation)
			self.layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, K, activation=T.nnet.softmax, use_bias=False)
		# layer = HiddenLayer(Mi, K, activation=lambda x: x, use_bias=False)
		self.layers.append(layer)

		# collect all params for gradient later
		params = []
		for layer in reversed(self.layers):
			params += layer.params
		caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in params]
		velocities = [theano.shared(p.get_value()*0) for p in params]

		# inputs and targets
		X = T.matrix('X')
		actions = T.ivector('actions')
		advantages = T.vector('advantages')

		# calculate outputs and cost
		Z = X
		for layer in self.layers:
			Z = layer.forward(Z)
		# action_scores = Z
		# p_a_given_s = T.nnet.softmax(action_scores)
		p_a_given_s = Z

		selected_probs = T.log(p_a_given_s[T.arange(actions.shape[0]), actions])
		cost = -T.sum(advantages * selected_probs)

		# specify update rule
		grads = T.grad(cost, params)
		c_update = [(c, decay*c + (1 - decay)*g*g) for c, g in zip(caches, grads)]
		v_update = [(v, mu*v - lr*g/T.sqrt(c)) for v, c, g in zip(velocities, caches, grads)]
		p_update = [(p, p + v) for p, v, g in zip(params, velocities, grads)]
		updates = c_update + v_update + p_update

		# compile functions
		self.train_op = theano.function(
			inputs=[X, actions, advantages],
			updates=updates,
			allow_input_downcast=True
		)

		self.predict_op = theano.function(
			inputs=[X],
			outputs=p_a_given_s,
			allow_input_downcast=True
		)

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		self.train_op(X, actions, advantages)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.predict_op(X)

	def sample_action(self, X):
		p = self.predict(X)[0]
		nonans = not np.any(np.isnan(p))
		assert(nonans)
		return np.random.choice(len(p), p=p)


# approximate V(s)
class ValueModel:
	def __init__(self, D, hidden_layer_sizes, activation=T.tanh, learning_rate=1e-4):
		# constant learning rate is fine
		lr = learning_rate

		# create the graph
		self.layers = []
		Mi = D
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=activation)
			self.layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, 1, activation=lambda x: x)
		self.layers.append(layer)

		# collect params for gradient later
		params = []
		for layer in reversed(self.layers):
			params += layer.params

		# inputs and targets
		X = T.matrix('X')
		Y = T.vector('Y')

		# calculate output and cost
		Z = X
		for layer in self.layers:
			Z = layer.forward(Z)
		Y_hat = T.flatten(Z)
		cost = T.sum((Y - Y_hat)**2)

		# specify update rule
		grads = T.grad(cost, params)
		updates = [(p, p - lr * g) for p, g in zip(params, grads)]

		# compile functions
		self.train_op = theano.function(
			inputs=[X, Y],
			updates=updates,
			allow_input_downcast=True
		)

		self.predict_op = theano.function(
			inputs=[X],
			outputs=Y_hat,
			allow_input_downcast=True
		)

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		Y = np.atleast_1d(Y)
		self.train_op(X, Y)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.predict_op(X)


def play_one_td(env, pmodel, vmodel, gamma, max_iters=3000):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < max_iters:
		action = pmodel.sample_action(observation)
		prev_observation = observation
		observation, reward, done, info = env.step(action)

		# if done:
		# 	reward = -200

		# update the model
		V_next = vmodel.predict(observation)
		G = reward + gamma * np.max(V_next) # np.max flatten V_next from (1, ) to ()
		advantage = G - vmodel.predict(prev_observation)
		pmodel.partial_fit(prev_observation, action, advantage)
		vmodel.partial_fit(prev_observation, G)

		if reward == 1: # if we changed the reward to -200 while done
			totalreward += reward
		iters += 1

	return totalreward


def play_one_mc(env, pmodel, vmodel, gamma, max_iters=3000):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	states = []
	actions = []
	rewards = []

	reward = 0
	while not done and iters < max_iters:
		action = pmodel.sample_action(observation)

		states.append(observation)
		actions.append(action)
		rewards.append(reward)

		prev_observation = observation
		observation, reward, done, info = env.step(action)

		# if done:
		# 	reward = -200

		if reward == 1: # if we changed the reward to -200 while done
			totalreward += reward
		iters += 1

	# save the final (s, a, r) tuple
	action = pmodel.sample_action(observation)
	states.append(observation)
	actions.append(action)
	rewards.append(reward)

	returns = []
	advantages = []
	G = 0
	for s, r in zip(reversed(states), reversed(rewards)):
		returns.append(G)
		advantages.append(G - vmodel.predict(s)[0])
		G = r + gamma * G
	returns.reverse()
	advantages.reverse()

	# update the model
	pmodel.partial_fit(states, actions, advantages)
	vmodel.partial_fit(states, returns)

	return totalreward


def main():
	# gym.envs.register(
	# 	id='MyCartPole-v0',
	# 	entry_point='gym.envs.classic_control:CartPoleEnv',
	# 	max_episode_steps=10000,
	# 	reward_threshold=9975.0,
	# )
	# env = gym.make('MyCartPole-v0')
	env = gym.make('CartPole-v0')

	# if 'monitor' in sys.argv:
	# 	filenema = os.path.basename(__file__).split('.')[0]
	# 	monitor_dir = './' + filenema + '_' + str(datetime.now())
	# 	env = wrappers.Monitor(env, monitor_dir)

	D = env.observation_space.shape[0]
	K = env.action_space.n

	pmodel = PolicyModel(D, K, hidden_layer_sizes=[10], activation=T.tanh, learning_rate=1e-3, momentum=0.8, decay=0.999)
	vmodel = ValueModel(D, hidden_layer_sizes=[30], activation=T.tanh, learning_rate=1e-4)
	gamma = 0.99

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		totalreward = play_one_mc(env, pmodel, vmodel, gamma)
		# totalreward = play_one_td(env, pmodel, vmodel, gamma)
		totalrewards[n] = totalreward
		if n % 20 == 0:
			print('episode: %d, current reward: %s, avg reward (last 100 episodes): %s' % (n, totalreward, totalrewards[max(0, n-99):(n+1)].mean()))

	print('avg reward for last 100 episodes: %s' % totalrewards[-100:].mean())
	print('total steps: %s' % totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)

	if 'monitor' in sys.argv:
		filenema = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filenema + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)
		play_one_mc(env, pmodel, vmodel, gamma)


if __name__ == '__main__':
	main()

