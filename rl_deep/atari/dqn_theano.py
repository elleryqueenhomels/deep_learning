# Use Deep Q-Network to solve Breakout

import os
import sys
import gym
import random
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from scipy.misc import imresize
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
STACK_FRAMES = 4
IMG_SIZE = 80


def downsample_image(org_img, img_size=IMG_SIZE):
	new_img = org_img[31:195] # select the important parts of the image
	new_img = new_img.mean(axis=2) # convert RGB to grayscale

	# downsample image
	# changing aspect ratio doesn't significantly distort the image
	# Nearest Neighbor Interpolation produces a much sharper image
	# than default bilinear interpolation
	new_img = imresize(new_img, size=(img_size, img_size), interp='nearest')
	return new_img


def update_state(state, observation):
	observation_small = downsample_image(observation)
	return np.append(state[1:], np.expand_dims(observation_small, 0), axis=0)


def init_filter(shape):
	W = np.random.randn(*shape) * 2 / np.sqrt(np.prod(shape[1:]))
	return W.astype(np.float32)


class ConvLayer:
	def __init__(self, mi, mo, filtersz=5, poolsz=2, activation=T.nnet.relu):
		# mi = num of input  feature maps
		# mo = num of output feature maps
		shape = (mo, mi, filtersz, filtersz)
		W0 = init_filter(shape)
		self.W = theano.shared(W0)
		b0 = np.zeros(mo, dtype=np.float32)
		self.b = theano.shared(b0)
		self.poolsz = poolsz
		self.params = [self.W, self.b]
		self.f = activation

	def forward(self, X):
		shape = (self.poolsz, self.poolsz)
		# without max_pool version
		# conv_out = conv2d(
		# 	input=X,
		# 	filters=self.W,
		# 	subsample=shape,
		# 	border_mode='half'
		# )
		# return self.f(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# with max_pool version
		conv_out = conv2d(input=X, filters=self.W)
		conv_out = self.f(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		conv_out = pool_2d(input=conv_out, ws=shape, ignore_border=True)
		return conv_out


class HiddenLayer:
	def __init__(self, Mi, Mo, activation=T.nnet.relu):
		W = np.random.randn(Mi, Mo) * np.sqrt(2 / Mi)
		self.W = theano.shared(W.astype(np.float32))
		self.b = theano.shared(np.zeros(Mo, dtype=np.float32))
		self.params = [self.W, self.b]
		self.f = activation

	def forward(self, X):
		a = X.dot(self.W) + self.b
		return self.f(a)


class DQN:
	def __init__(self, D, K, conv_layer_sizes, hidden_layer_sizes, activation_conv=T.nnet.relu, activation_hidden=T.nnet.relu):
		lr = np.float32(2.5e-4)
		mu = np.float32(0)
		decay = np.float32(0.99)
		eps = np.float32(1e-10)
		one = np.float32(1)
		self.K = K
		# D == (STACK_FRAMES, IMG_SIZE, IMG_SIZE)

		# inputs and targets
		X = T.ftensor4('X')
		G = T.fvector('G')
		actions = T.ivector('actions')

		# create the graph
		self.conv_layers = []
		num_input_filters = D[0]
		for num_output_filters, filtersz, poolsz in conv_layer_sizes:
			layer = ConvLayer(num_input_filters, num_output_filters, filtersz, poolsz, activation=activation_conv)
			self.conv_layers.append(layer)
			num_input_filters = num_output_filters

		# get conv output size
		Z = X / 255.0
		for layer in self.conv_layers:
			Z = layer.forward(Z)
		conv_out = Z.flatten(ndim=2)
		conv_out_op = theano.function(inputs=[X], outputs=conv_out, allow_input_downcast=True)
		test = conv_out_op(np.random.randn(1, *D))
		flattened_output_size = test.shape[1]

		# build fully-connected layers
		self.hidden_layers = []
		Mi = flattened_output_size
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=activation_hidden)
			self.hidden_layers.append(layer)
			Mi = Mo

		# final layer
		# layer = HiddenLayer(Mi, K, activation=T.identity)
		layer = HiddenLayer(Mi, K, activation=lambda x: x)
		self.hidden_layers.append(layer)

		# collect params for copy
		self.params = []
		for layer in (self.conv_layers + self.hidden_layers):
			self.params += layer.params

		# calculate final output and cost
		Z = conv_out
		for layer in self.hidden_layers:
			Z = layer.forward(Z)
		Y_hat = Z

		selected_action_values = Y_hat[T.arange(actions.shape[0]), actions]
		cost = T.sum((G - selected_action_values)**2)

		# create train function
		# we need to ensure cache is updated before parameter update
		# by creating a list of new_caches
		# using them in the parameter update
		grads = T.grad(cost, self.params)
		caches = [theano.shared(np.ones_like(p.get_value())*0.1) for p in self.params]
		velocities = [theano.shared(p.get_value()*0) for p in self.params]

		c_update = [(c, decay*c + (one - decay)*g*g) for c, g in zip(caches, grads)]
		v_update = [(v, mu*v - lr*g / T.sqrt(c + eps)) for v, c, g in zip(velocities, caches, grads)]
		p_update = [(p, p + v) for p, v in zip(self.params, velocities)]
		updates  = c_update + v_update + p_update

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

	def copy_from(self, other):
		for p, q in zip(self.params, other.params):
			v = q.get_value()
			p.set_value(v)

	def predict(self, X):
		return self.predict_op(X)

	def update(self, states, actions, targets):
		self.train_op(states, targets, actions)

	def sample_action(self, x, eps):
		if np.random.random() < eps:
			return np.random.choice(self.K)
		else:
			return np.argmax(self.predict([x])[0])


def train(model, target_model, experience_replay_buffer, gamma, batch_size):
	# Sample experiences
	samples = random.sample(experience_replay_buffer, batch_size)
	states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

	# Calculate targets
	next_Qs = target_model.predict(next_states)
	next_Q  = np.max(next_Qs, axis=1)
	flags = np.invert(dones).astype(np.float32)
	targets = rewards + flags * gamma * next_Q

	# Update the model
	model.update(states, actions, targets)


def initial_state(env):
	observation = env.reset()
	observation_small = downsample_image(observation)
	# state = np.array([observation_small] * STACK_FRAMES)
	state = np.stack([observation_small] * STACK_FRAMES, axis=0)
	return state


def play_one(env, total_steps, experience_replay_buffer, model, target_model, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period=TARGET_UPDATE_PERIOD):
	t0 = datetime.now()

	# reset the environment
	state = initial_state(env)
	assert(state.shape == (STACK_FRAMES, IMG_SIZE, IMG_SIZE))

	total_time_training = 0
	num_steps_in_episode = 0
	episode_reward = 0

	done = False
	while not done:
		# Update target network
		if total_steps % target_update_period == 0:
			target_model.copy_from(model)
			print('Copied model parameters to target network. total_steps=%s, update_period=%s' % (total_steps, target_update_period))

		# Take action
		action = model.sample_action(state, epsilon)
		observation, reward, done, info = env.step(action)
		next_state = update_state(state, observation)

		episode_reward += reward

		# Remove oldest experience if replay buffer is full
		if len(experience_replay_buffer) == MAX_EXPERIENCES:
			experience_replay_buffer.pop(0)
			print('Experience Replay Buffer is full. Pop out the first element.')

		# Save the latest experience
		experience_replay_buffer.append((state, action, reward, next_state, done))

		# Train the model, keep track of time
		t0_2 = datetime.now()
		train(model, target_model, experience_replay_buffer, gamma, batch_size)
		dt = datetime.now() - t0_2

		total_time_training += dt.total_seconds()
		num_steps_in_episode += 1

		state = next_state
		total_steps += 1

		epsilon = max(epsilon - epsilon_change, epsilon_min)

	duration = datetime.now() - t0
	training_time_per_step = total_time_training / num_steps_in_episode

	return total_steps, episode_reward, duration, num_steps_in_episode, training_time_per_step, epsilon


def main():
	# hyperparameters and initialize stuff
	# conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)] # without max_pool version
	conv_layer_sizes = [(32, 5, 2), (64, 5, 2), (64, 3, 2)] # with max_pool version
	hidden_layer_sizes = [512]
	gamma = 0.99
	batch_size = 32
	num_episodes = 10000
	total_steps = 0
	experience_replay_buffer = []
	episode_rewards = np.zeros(num_episodes)

	# epsilon
	# decays linearly until 0.1
	epsilon = 1.0
	epsilon_min = 0.1
	epsilon_change = (epsilon - epsilon_min) / 500000

	# Create environment
	env = gym.make('Breakout-v0')

	D = (STACK_FRAMES, IMG_SIZE, IMG_SIZE)
	K = env.action_space.n

	# Create models
	model = DQN(
		D=D,
		K=K,
		conv_layer_sizes=conv_layer_sizes,
		hidden_layer_sizes=hidden_layer_sizes,
		activation_conv=T.nnet.relu,
		activation_hidden=T.nnet.relu
	)
	target_model = DQN(
		D=D,
		K=K,
		conv_layer_sizes=conv_layer_sizes,
		hidden_layer_sizes=hidden_layer_sizes,
		activation_conv=T.nnet.relu,
		activation_hidden=T.nnet.relu
	)

	print('Generating Experience Replay Buffer...')
	state = initial_state(env)
	for i in range(MIN_EXPERIENCES):
		action = np.random.choice(K)
		observation, reward, done, info = env.step(action)
		next_state = update_state(state, observation)
		experience_replay_buffer.append((state, action, reward, next_state, done))

		if done:
			state = initial_state(env)
		else:
			state = next_state

	# Play a number of episodes and train
	print('Begin to play episodes and train...')
	for i in range(num_episodes):
		total_steps, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
			env,
			total_steps,
			experience_replay_buffer,
			model,
			target_model,
			gamma,
			batch_size,
			epsilon,
			epsilon_change,
			epsilon_min
		)
		episode_rewards[i] = episode_reward

		last_100_avg = episode_rewards[max(0, i - 99):(i + 1)].mean()

		print(
			'Episode: %d,' % i,
			'Duration: %s,' % duration,
			'Num steps in episode: %d,' % num_steps_in_episode,
			'Episode reward: %s,' % episode_reward,
			'Training time per step: %.3f sec,' % time_per_step,
			'Avg Reward (Last 100): %.3f,' % last_100_avg,
			'Epsilon: %.6f' % epsilon
		)
		sys.stdout.flush()

	plt.plot(episode_rewards)
	plt.title('Rewards / Episodes')
	plt.show()

	if 'monitor' in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)
		play_one(
			env,
			total_steps,
			experience_replay_buffer,
			model,
			target_model,
			gamma,
			batch_size,
			0,
			0,
			0
		)


if __name__ == '__main__':
	main()

