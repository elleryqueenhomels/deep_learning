# Use Hill Climbing to solve MountainCarContinuous

import os
import sys
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer, plot_running_avg


class HiddenLayer:
	def __init__(self, Mi, Mo, activation=tf.tanh, use_bias=True, zeros=False):
		if zeros:
			W = np.zeros((Mi, Mo)).astype(np.float32)
			self.W = tf.Variable(W)
		else:
			self.W = tf.Variable(tf.random_normal(shape=(Mi, Mo)))
		self.params = [self.W]
		self.use_bias = use_bias
		if use_bias:
			self.b = tf.Variable(np.zeros(Mo).astype(np.float32))
			self.params.append(self.b)
		self.f = activation

	def forward(self, X):
		if self.use_bias:
			a = tf.matmul(X, self.W) + self.b
		else:
			a = tf.matmul(X, self.W)
		return self.f(a)


# approximate pi(a | s)
class PolicyModel:
	def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[], activation=tf.tanh):
		# save inputs for copy
		self.ft = ft
		self.D = D
		self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
		self.hidden_layer_sizes_var = hidden_layer_sizes_var
		self.activation = activation

		##### model the mean #####
		self.mean_layers = []
		Mi = D
		for Mo in hidden_layer_sizes_mean:
			layer = HiddenLayer(Mi, Mo, activation=activation)
			self.mean_layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, 1, activation=lambda x: x, use_bias=False, zeros=True)
		self.mean_layers.append(layer)


		##### model the variance #####
		self.var_layers = []
		Mi = D
		for Mo in hidden_layer_sizes_var:
			layer = HiddenLayer(Mi, Mo, activation=activation)
			self.var_layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, 1, activation=tf.nn.softplus, use_bias=False, zeros=False)
		self.var_layers.append(layer)

		# gather params
		self.params = []
		for layer in (self.mean_layers + self.var_layers):
			self.params += layer.params

		# inputs and targets
		self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
		self.actions = tf.placeholder(tf.float32, shape=(None, ), name='actions')
		self.advantages = tf.placeholder(tf.float32, shape=(None, ), name='advantages')

		def get_out(layers):
			Z = self.X
			for layer in layers:
				Z = layer.forward(Z)
			return tf.reshape(Z, [-1])

		# calculate output and cost
		mean = get_out(self.mean_layers)
		var = get_out(self.var_layers) + 1e-4 # smoothing

		# log_probs = log_pdf(self.actions, mean, var)
		norm = tf.contrib.distributions.Normal(mean, var)
		self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

		log_probs = norm.log_prob(self.actions)
		cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*tf.log(2*np.pi*var)) + 0.1*tf.reduce_sum(mean*mean)
		self.cost = cost
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
		# self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
		# self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
		# self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

	def set_session(self, session):
		self.session = session

	def init_vars(self):
		init_op = tf.variables_initializer(self.params)
		self.session.run(init_op)

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		if self.ft is not None:
			X = self.ft.transform(X)
		self.session.run(
			self.train_op,
			feed_dict={
				self.X: X,
				self.actions: actions,
				self.advantages: advantages,
			}
		)

	def predict(self, X):
		X = np.atleast_2d(X)
		if self.ft is not None:
			X = self.ft.transform(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return p

	def copy(self):
		clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var, self.activation)
		clone.set_session(self.session)
		clone.init_vars()
		clone.copy_from(self)
		return clone

	def copy_from(self, other):
		# collect all the ops
		ops = []
		for p, q in zip(self.params, other.params):
			v = self.session.run(q)
			op = p.assign(v)
			ops.append(op)
		self.session.run(ops)

	def perturb_params(self, noise_magnitude=5.0, eps=0.1):
		ops = []
		for p in self.params:
			v = self.session.run(p)
			noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * noise_magnitude
			if np.random.random() < eps:
				# with probability eps start completely from scratch
				op = p.assign(noise)
			else:
				op = p.assign(v + noise)
			ops.append(op)
		self.session.run(ops)


def play_one(env, pmodel, gamma, max_iters=2000):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < max_iters:
		action = pmodel.sample_action(observation)
		# oddly, the mountain car environment requires the action to be in
		# an object where actual action is stored in object[0]
		observation, reward, done, info = env.step([action])

		totalreward += reward
		iters += 1

	return totalreward


def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False):
	totalrewards = np.empty(T)

	for t in range(T):
		totalrewards[t] = play_one(env, pmodel, gamma)

		if print_iters:
			print('episode: %d, avg reward so far: %s' % (t, totalrewards[:(t+1)].mean()))

	avg_totalrewards = totalrewards.mean()
	print('avg totalrewards for all %d episodes: %s' % (T, avg_totalrewards))
	return avg_totalrewards


def random_search(env, pmodel, gamma, num_searchs=100, num_episodes_per_param_test=3):
	totalrewards = []
	best_avg_totalreward = float('-inf')
	best_pmodel = pmodel

	for t in range(num_searchs):
		tmp_pmodel = best_pmodel.copy()

		tmp_pmodel.perturb_params()

		avg_totalrewards = play_multiple_episodes(
			env,
			num_episodes_per_param_test,
			tmp_pmodel,
			gamma
		)

		totalrewards.append(avg_totalrewards)

		if avg_totalrewards > best_avg_totalreward:
			best_avg_totalreward = avg_totalrewards
			best_pmodel = tmp_pmodel

	return totalrewards, best_pmodel


def main():
	# gym.envs.register(
	# 	id='MyMountainCarContinuous-v0',
	# 	entry_point='gym.envs.classic_control:MountainCarContinuousEnv',
	# 	max_episode_steps=10000,
	# 	reward_threshold=90.0,
	# )
	# env = gym.make('MyMountainCarContinuous-v0')
	env = gym.make('MountainCarContinuous-v0')
	ft = FeatureTransformer(env, n_components=100, n_samples=10000)
	D = ft.dimensions
	pmodel = PolicyModel(ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[], activation=tf.tanh)
	session = tf.InteractiveSession()
	pmodel.set_session(session)
	pmodel.init_vars()
	gamma = 0.99

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	totalrewards, pmodel = random_search(env, pmodel, gamma)
	print('max reward: %s' % np.max(totalrewards))

	# play 100 episodes and check the average
	avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
	print('avg reward for 100 episodes with best model: %s' % avg_totalrewards)

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()


if __name__ == '__main__':
	main()

