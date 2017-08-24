# Use DQN (Deep Q-Network) to solve MountainCar

import gym
import time
import random
import numpy as np
import tensorflow as tf


# ------------------ Constants ------------------
ENV = 'MountainCar-v0'

HIDDEN_LAYER_SIZES = [72, 36, 6]

GAMMA = 0.99
BATCH_SZ = 32
MEMORY_CAPACITY = 100000

LEARNING_RATE = 1e-3

EPS_START = 0.4
EPS_STOP  = 0.1
EPS_STEPS = 10000

NUM_EPISODES = 500


# ------------------ Classes ------------------
# Uniform Experience Replay Memory
class ExperienceReplayMemory:

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []

	def current_length(self):
		return len(self.memory)

	def push(self, event):
		self.memory.append(event)
		if len(self.memory) > self.capacity:
			self.memory.pop(0)

	def sample(self, batch_sz):
		samples = random.sample(self.memory, batch_sz)
		return map(np.array, zip(*samples))


# Create the Deep Q-Network
class DQN:

	def __init__(self, input_sz, output_sz, hidden_layer_sizes, gamma, batch_sz, memory_capacity, learning_rate=LEARNING_RATE, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		self.input_sz = input_sz
		self.output_sz = output_sz
		self.gamma = gamma
		self.batch_sz = batch_sz
		self.memory = ExperienceReplayMemory(memory_capacity)

		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps
		self.eps_delta = (eps_end - eps_start) / eps_steps
		self.step_count = 0

		# inputs and targets
		self.states  = tf.placeholder(tf.float32, shape=(None, input_sz), name='states')
		self.actions = tf.placeholder(tf.int32, shape=(None, ), name='actions')
		self.targets = tf.placeholder(tf.float32, shape=(None, ), name='targets')

		# create the graph
		Z = self.states
		for M in hidden_layer_sizes:
			Z = tf.contrib.layers.fully_connected(Z, M, activation_fn=tf.nn.relu)

		# final output layer
		Q_values = tf.contrib.layers.fully_connected(Z, output_sz, activation_fn=lambda x: x)
		self.predict_op = Q_values

		selected_action_values = tf.reduce_sum(Q_values * tf.one_hot(self.actions, output_sz), axis=1)

		# An alternative method to calculate selected_action_values:
		# we would like to do this, but it doesn't work in TensorFlow:
		# selected_action_values = Q_values[tf.range(batch_sz), self.actions]
		# instead we do:
		# indices = tf.range(tf.shape(Q_values)[0]) * tf.shape(Q_values)[1] + self.actions
		# selected_action_values = tf.gather(
		# 	tf.reshape(Q_values, [-1]),
		# 	indices
		# )

		# cost = tf.reduce_sum(tf.square(self.targets - selected_action_values)) # may use tf.reduce_mean()
		cost = tf.nn.l2_loss(self.targets - selected_action_values)

		self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		# self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(cost)

		self.params = tf.trainable_variables() # collect the model params

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

	def set_session(self, session):
		self.session = session

	def predict(self, states):
		states = np.atleast_2d(states)
		return self.session.run(self.predict_op, feed_dict={self.states: states})

	def get_epsilon(self):
		if self.step_count >= self.eps_steps:
			return self.eps_end
		else:
			return self.eps_start + self.step_count * self.eps_delta # linearly interpolates

	def select_action(self, state):
		eps = self.get_epsilon()
		self.step_count += 1

		if np.random.random() < eps:
			return np.random.choice(self.output_sz)
		else:
			state = np.atleast_2d(state)
			return np.argmax(self.predict(state)[0])

	def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
		next_outputs = self.predict(batch_next_states).max(axis=1)
		done_masks = np.invert(batch_dones).astype(np.float32)
		targets = batch_rewards + self.gamma * next_outputs * done_masks

		self.session.run(self.train_op, feed_dict={self.states: batch_states, self.actions: batch_actions, self.targets: targets})

	def update(self, state, action, reward, next_state, done):
		event = (state, action, reward, next_state, done)
		self.memory.push(event)

		if self.memory.current_length() > self.batch_sz:
			batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.memory.sample(self.batch_sz)
			self.learn(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)


# ------------------ Entry Point ------------------
def play_one_episode(env, model, render=False):
	s = env.reset()

	done = False
	total_reward = 0
	while not done:
		if render:
			env.render()

		a = model.select_action(s)
		s2, r, done, info = env.step(a)

		model.update(s, a, r, s2, done)

		s = s2
		total_reward += r

	return total_reward

if __name__ == '__main__':
	env = gym.make(ENV)

	num_state   = env.observation_space.shape[0]
	num_actions = env.action_space.n

	model = DQN(num_state, num_actions, HIDDEN_LAYER_SIZES, GAMMA, BATCH_SZ, MEMORY_CAPACITY)

	total_rewards = np.zeros(NUM_EPISODES)
	for n in range(NUM_EPISODES):
		total_reward = play_one_episode(env, model)
		total_rewards[n] = total_reward

		if n % 10 == 0:
			print('episode: %d, current reward: %s, last 100 episodes avg reward: %s' % (n, total_reward, total_rewards[max(0, n-99):(n+1)].mean()))

	print('avg reward for last 100 episodes: %s' % total_rewards[-100:].mean())

	# test
	model.eps_start = 0.
	model.eps_delta = 0.
	model.step_count = 0
	while True:
		time.sleep(0.01)
		total_reward = play_one_episode(env, model, render=True)
		print('Reward: %s' % total_reward)

