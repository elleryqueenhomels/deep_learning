# Use Deep Q-Network to solve CartPole

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


# approximate Q(s, a) for all a, i.e. input s (shape=[N, D]), output Q (shape=[N, K])
class DQN:
	def __init__(self, D, K, hidden_layer_sizes, gamma, activation=T.tanh, max_experiences=10000, min_experiences=100, batch_sz=32):
		lr = 1e-2
		mu = 0.
		decay = 0.99
		self.K = K

		# create the graph
		self.layers = []
		Mi = D
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=activation)
			self.layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, K, activation=lambda x: x)
		self.layers.append(layer)

		# collect params for copy
		self.params = []
		for layer in self.layers:
			self.params += layer.params
		caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in self.params]
		velocities = [theano.shared(p.get_value()*0) for p in self.params]

		# inputs and targets
		X = T.matrix('X')
		G = T.vector('G')
		actions = T.ivector('actions')

		# calculate output and cost
		Z = X
		for layer in self.layers:
			Z = layer.forward(Z)
		Y_hat = Z

		selected_action_values = Y_hat[T.arange(actions.shape[0]), actions]
		cost = T.sum((G - selected_action_values)**2)

		# create train function
		grads = T.grad(cost, self.params)
		c_update = [(c, decay*c + (1 - decay)*g*g) for c, g in zip(caches, grads)]
		v_update = [(v, mu*v - lr*g / T.sqrt(c)) for v, c, g in zip(velocities, caches, grads)]
		p_update = [(p, p + v) for p, v in zip(self.params, velocities)]
		updates = c_update + v_update + p_update

		# compile functions
		self.train_op = theano.function(
			inputs=[X, G, actions],
			updates=updates,
			allow_input_downcast=True
		)

		self.predict_op = theano.function(
			inputs=[X],
			outputs=Y_hat,
			allow_input_downcast=True
		)

		# create replay memory
		self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
		self.max_experiences = max_experiences
		self.min_experiences = min_experiences
		self.batch_sz = batch_sz
		self.gamma = gamma

	def copy_from(self, other):
		for p, q in zip(self.params, other.params):
			v = q.get_value()
			p.set_value(v)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.predict_op(X)

	def train(self, target_network):
		# sample a random batch from buffer, do an iteration of GD
		if len(self.experience['s']) < self.min_experiences:
			# don't do anything if we don't have enough experience
			return

		# randomly select a batch
		idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
		states = [self.experience['s'][i] for i in idx]
		actions = [self.experience['a'][i] for i in idx]
		rewards = [self.experience['r'][i] for i in idx]
		next_states = [self.experience['s2'][i] for i in idx]
		dones = [self.experience['done'][i] for i in idx]
		next_Q = np.max(target_network.predict(next_states), axis=1)
		targets = [r + self.gamma * next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

		# call optimizer
		self.train_op(states, targets, actions)

	def add_experience(self, s, a, r, s2, done):
		if len(self.experience['s']) >= self.max_experiences:
			self.experience['s'].pop(0)
			self.experience['a'].pop(0)
			self.experience['r'].pop(0)
			self.experience['s2'].pop(0)
			self.experience['done'].pop(0)
		self.experience['s'].append(s)
		self.experience['a'].append(a)
		self.experience['r'].append(r)
		self.experience['s2'].append(s2)
		self.experience['done'].append(done)

	def sample_action(self, x, eps):
		if np.random.random() < eps:
			return np.random.choice(self.K)
		else:
			X = np.atleast_2d(x)
			return np.argmax(self.predict(X)[0])


def play_one(env, model, tmodel, eps, gamma, copy_period, max_iters=2000):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < max_iters:
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = env.step(action)

		totalreward += reward
		# if done:
		# 	reward = -200

		# update the model
		model.add_experience(prev_observation, action, reward, observation, done)
		model.train(tmodel)

		iters += 1

		if iters % copy_period == 0:
			tmodel.copy_from(model)

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

	gamma = 0.99
	copy_period = 50

	D = len(env.observation_space.sample())
	K = env.action_space.n
	hidden_layer_sizes = [200, 200]
	model  = DQN(D, K, hidden_layer_sizes, gamma, activation=T.tanh)
	tmodel = DQN(D, K, hidden_layer_sizes, gamma, activation=T.tanh)
	tmodel.copy_from(model)

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1.0 / np.sqrt(n + 1)
		totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
		totalrewards[n] = totalreward
		if n % 20 == 0:
			print('episode: %d, current reward: %s, eps: %s, avg reward (last 100): %s' % (n, totalreward, eps, totalrewards[max(0, n-99):(n+1)].mean()))

	print('avg reward for last 100 episodes: %s' % totalrewards[-100:].mean())
	print('total steps: %s' % totalrewards.sum())

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()

	plot_running_avg(totalrewards)

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)
		play_one(env, model, tmodel, eps, gamma, copy_period)


if __name__ == '__main__':
	main()

