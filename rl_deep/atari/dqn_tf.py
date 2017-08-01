# Use Deep Q-Network to solve Atari-Breakout

import os
import sys
import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gym import wrappers
from datetime import datetime
from scipy.misc import imresize


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


class DQN:
	def __init__(self, D, K, conv_layer_sizes, hidden_layer_sizes, scope, activation_conv=tf.nn.relu, activation_hidden=tf.nn.relu):

		self.K = K
		self.scope = scope

		with tf.variable_scope(scope):

			# inputs and targets
			# D == (STACK_FRAMES, IMG_SIZE, IMG_SIZE)
			# self.X = tf.placeholder(tf.float32, shape=(None, STACK_FRAMES, IMG_SIZE, IMG_SIZE), name='X')
			self.X = tf.placeholder(tf.float32, shape=(None, *D), name='X')
			# tensorflow convolution needs the order to be:
			# (num_samples, height, width, channels)
			# so we need to transpose later
			self.G = tf.placeholder(tf.float32, shape=(None, ), name='G')
			self.actions = tf.placeholder(tf.int32, shape=(None, ), name='actions')

			# calculate output and cost
			# convolutional layers
			# these built-in layers are faster and don't require us to
			# calculate the size of the output of the final conv layer
			Z = self.X / 255.0
			Z = tf.transpose(Z, [0, 2, 3, 1])
			for num_output_filters, filtersz, poolsz in conv_layer_sizes:
				Z = tf.contrib.layers.conv2d(
					Z,
					num_output_filters,
					filtersz,
					poolsz,
					activation_fn=activation_conv
				)

			# fully-connected layers
			Z = tf.contrib.layers.flatten(Z)
			for M in hidden_layer_sizes:
				Z = tf.contrib.layers.fully_connected(Z, M, activation_fn=activation_hidden)

			# final output layer
			self.predict_op = tf.contrib.layers.fully_connected(Z, K, activation_fn=tf.identity)

			selected_action_values = tf.reduce_sum(
				self.predict_op * tf.one_hot(self.actions, K),
				reduction_indices=[1]
			)

			cost = tf.reduce_mean(tf.square(self.G - selected_action_values))

			self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
			# self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(cost)
			# self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
			# self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
			# self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

			self.cost = cost

	def copy_from(self, other):
		self_params = [v for v in tf.trainable_variables() if v.name.startswith(self.scope)]
		self_params = sorted(self_params, key=lambda v: v.name)
		other_params = [v for v in tf.trainable_variables() if v.name.startswith(other.scope)]
		other_params = sorted(other_params, key=lambda v: v.name)

		ops = []
		for p, q in zip(self_params, other_params):
			v = self.session.run(q)
			op = p.assign(v)
			ops.append(op)

		self.session.run(ops)

	def set_session(self, session):
		self.session = session

	def predict(self, states):
		return self.session.run(self.predict_op, feed_dict={self.X: states})

	def update(self, states, actions, targets):
		c, _ = self.session.run(
			[self.cost, self.train_op],
			feed_dict={
				self.X: states,
				self.G: targets,
				self.actions: actions
			}
		)
		return c

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

	# Update model
	loss = model.update(states, actions, targets)
	return loss


def play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period=TARGET_UPDATE_PERIOD):
	t0 = datetime.now()

	# reset the environment
	observation = env.reset()
	observation_small = downsample_image(observation)
	state = np.stack([observation_small] * STACK_FRAMES, axis=0)
	assert(state.shape == (STACK_FRAMES, IMG_SIZE, IMG_SIZE))
	loss = None

	total_time_training = 0
	num_steps_in_episode = 0
	episode_reward = 0

	done = False

	while not done:
		# Update target network
		if total_t % target_update_period == 0:
			target_model.copy_from(model)
			print('Copied model parameters to target network. total_t = %s, period = %s' % (total_t, target_update_period))

		# Take action
		action = model.sample_action(state, epsilon)
		observation, reward, done, info = env.step(action)
		next_state = update_state(state, observation)

		episode_reward += reward

		# Remove oldest experience if replay buffer is full
		if len(experience_replay_buffer) == MAX_EXPERIENCES:
			experience_replay_buffer.pop(0)

		# Save the latest experience
		experience_replay_buffer.append((state, action, reward, next_state, done))

		# Train the model, keep track of time
		t0_2 = datetime.now()
		loss = train(model, target_model, experience_replay_buffer, gamma, batch_size)
		dt = datetime.now() - t0_2

		total_time_training += dt.total_seconds()
		num_steps_in_episode += 1

		state = next_state
		total_t += 1

		epsilon = max(epsilon - epsilon_change, epsilon_min)

	return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon


def main():
	# hyperparameters and initialize stuff
	conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
	hidden_layer_sizes = [512]
	gamma = 0.99
	batch_sz = 32
	num_episodes = 10000
	total_t = 0
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
		scope='model',
		activation_conv=tf.nn.relu,
		activation_hidden=tf.nn.relu
	)
	target_model = DQN(
		D=D,
		K=K,
		conv_layer_sizes=conv_layer_sizes,
		hidden_layer_sizes=hidden_layer_sizes,
		scope='target_model',
		activation_conv=tf.nn.relu,
		activation_hidden=tf.nn.relu
	)

	with tf.Session() as sess:
		model.set_session(sess)
		target_model.set_session(sess)
		sess.run(tf.global_variables_initializer())

		print('Populating Experience Replay Buffer...')
		observation = env.reset()
		observation_small = downsample_image(observation)
		state = np.stack([observation_small] * STACK_FRAMES, axis=0)
		for i in range(MIN_EXPERIENCES):
			action = np.random.choice(K)
			observation, reward, done, info = env.step(action)
			next_state = update_state(state, observation)
			experience_replay_buffer.append((state, action, reward, next_state, done))

			if done:
				observation = env.reset()
				observation_small = downsample_image(observation)
				state = np.stack([observation_small] * STACK_FRAMES, axis=0)
			else:
				state = next_state

		# Play a number of episodes and train!
		print('Begin to play episodes and train...')
		for i in range(num_episodes):
			total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
				env,
				total_t,
				experience_replay_buffer,
				model,
				target_model,
				gamma,
				batch_sz,
				epsilon,
				epsilon_change,
				epsilon_min
			)
			episode_rewards[i] = episode_reward

			last_100_avg = episode_rewards[max(0, i-99):(i+1)].mean()

			print(
				'Episode:', i,
				'Duration:', duration,
				'Num steps:', num_steps_in_episode,
				'Reward:', episode_reward,
				'Training time per step: %.3f' % time_per_step,
				'Avg Reward (Last 100): %.3f' % last_100_avg,
				'Epsilon: %.3f' % epsilon
			)
			sys.stdout.flush()

		if 'monitor' in sys.argv:
			filename = os.path.basename(__file__).split('.')[0]
			monitor_dir = './' + filename + '_' + str(datetime.now())
			env = wrappers.Monitor(env, monitor_dir)
			play_one(
				env,
				total_t,
				experience_replay_buffer,
				model,
				target_model,
				gamma,
				batch_sz,
				epsilon,
				epsilon_change,
				epsilon_min
			)


if __name__ == '__main__':
	main()

