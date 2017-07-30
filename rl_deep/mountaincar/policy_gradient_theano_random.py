# Use Hill Climbing to solve MountainCarContinuous

import os
import sys
import gym
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer, plot_running_avg


class HiddenLayer:
	def __init__(self, Mi, Mo, activation=T.tanh, use_bias=True, zeros=False):
		if zeros:
			W = np.zeros((Mi, Mo))
		else:
			W = np.random.randn(Mi, Mo) * np.sqrt(2 / Mi)
		self.W = theano.shared(W)
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
	def __init__(self, ft, D, hidden_layer_sizes_mean, hidden_layer_sizes_var, activation=T.tanh, learning_rate=1e-2, momentum=0, decay=0.999):
		# starting learning rate and other hyperparams
		self.lr = learning_rate
		self.mu = momentum
		self.decay = decay

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
		layer = HiddenLayer(Mi, 1, activation=T.nnet.softplus, use_bias=False, zeros=False)
		self.var_layers.append(layer)

		# collect all params for gradient later
		params = []
		for layer in (self.mean_layers + self.var_layers):
			params += layer.params
		caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in params]
		velocities = [theano.shared(p.get_value()*0) for p in params]
		self.params = params

		# inputs and targets
		X = T.matrix('X')
		actions = T.vector('actions')
		advantages = T.vector('advantages')

		# calculate output and cost
		def get_output(layers):
			Z = X
			for layer in layers:
				Z = layer.forward(Z)
			return Z.flatten()

		mean = get_output(self.mean_layers)
		var = get_output(self.var_layers) + 1e-4 # smoothing

		# can't find Theano log pdf, we will make it
		def log_pdf(actions, mean, var):
			k1 = T.log(2 * np.pi * var)
			k2 = (actions - mean)**2 / var
			return -0.5*(k1 + k2)

		log_probs = log_pdf(actions, mean, var)
		cost = -T.sum(advantages * log_probs + 0.1 * T.log(2*np.pi*var)) + 0.1 * mean.dot(mean)

		self.get_log_probs = theano.function(
			inputs=[X, actions],
			outputs=log_probs,
			allow_input_downcast=True
		)

		# specify update rule
		grads = T.grad(cost, params)
		c_update = [(c, self.decay*c + (1 - self.decay)*g*g) for c, g in zip(caches, grads)]
		v_update = [(v, self.mu*v - self.lr*g / T.sqrt(c)) for v, c, g in zip(velocities, caches, grads)]
		p_update = [(p, p + v) for p, v in zip(params, velocities)]
		updates = c_update + v_update + p_update

		# compile functions
		self.train_op = theano.function(
			inputs=[X, actions, advantages],
			updates=updates,
			allow_input_downcast=True
		)

		# alternatively, we could create a RandomStream and sample from
		# the Gaussian using Theano code
		self.predict_op = theano.function(
			inputs=[X],
			outputs=[mean, var],
			allow_input_downcast=True
		)

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		if self.ft is not None:
			X = self.ft.transform(X)
		# log_probs = self.get_log_probs(X, actions)
		# print('log_probs.shape:', log_probs.shape)
		self.train_op(X, actions, advantages)

	def predict(self, X):
		X = np.atleast_2d(X)
		if self.ft is not None:
			X = self.ft.transform(X)
		return self.predict_op(X)

	def sample_action(self, X, clamp=(-1, 1)):
		pred = self.predict(X)
		mu = pred[0][0]
		v  = pred[1][0]
		a  = np.random.randn()*np.sqrt(v) + mu
		if clamp is None or len(clamp) < 2:
			return a
		else:
			return min(max(a, clamp[0]), clamp[1])

	def copy(self):
		clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var, self.activation, self.lr, self.mu, self.decay)
		clone.copy_from(self)
		return clone

	def copy_from(self, other):
		# self is being copy from other
		for p, q in zip(self.params, other.params):
			v = q.get_value()
			p.set_value(v)

	def perturb_params(self, noise_magnitude=5.0, eps=0.1):
		for p in self.params:
			v = p.get_value()
			noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * noise_magnitude
			if np.random.random() < eps:
				# with probability eps start completely from scratch
				p.set_value(noise)
			else:
				p.set_value(v + noise)


def play_one(env, pmodel, gamma, max_iters=2000):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < max_iters:
		action = pmodel.sample_action(observation)
		# oddly, the mountain car environment requires the action to be in
		# an object where the actual action is stored in object[0]
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
	pmodel = PolicyModel(ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[], activation=T.tanh)
	gamma = 0.99

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)

	totalrewards, pmodel = random_search(env, pmodel, gamma, num_searchs=100)

	print('max reward: %s' % np.max(totalrewards))

	# play 100 episodes and check the average
	avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
	print('avg reward over 100 episodes with best models: %s' % avg_totalrewards)

	plt.plot(totalrewards)
	plt.title('Rewards')
	plt.show()


if __name__ == '__main__':
	main()

