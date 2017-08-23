# Use A3C (Asynchronous Advantage Actor-Critic) to solve CartPole
# A3C implementation with GPU optimizer threads

import gym
import time
import threading
import numpy as np
import tensorflow as tf


# ----- Constants -----
ENV = 'CartPole-v0'

RUN_TIME = 30 # seconds
NUM_ENV_THREADS = 8
NUM_OPTIMIZERS  = 2
THREAD_DELAY = 0.001 # second

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = 0.1
EPS_STEPS = 20000

MIN_BATCH_SZ = 32
MAX_BATCH_SZ = MIN_BATCH_SZ * 5
LEARNING_RATE = 5e-3

LOSS_VALUE = 0.5 # value loss coefficient
LOSS_ENTROPY = 0.01 # entropy coefficient


class Brain:

	def __init__(self, num_state, num_actions, hidden_layer_sizes):
		self.train_queue = [[], [], [], [], []] # s, a, r, s', s'_terminal_mask
		self.lock_queue = threading.Lock()

		# placeholders
		self.states  = tf.placeholder(tf.float32, shape=(None, num_state), name='states')
		self.actions = tf.placeholder(tf.int32,   shape=(None, ), name='actions')
		self.returns = tf.placeholder(tf.float32, shape=(None, ), name='returns') # discounted n-step return

		# build the graph
		Z = self.states
		for M in hidden_layer_sizes:
			Z = tf.contrib.layers.fully_connected(Z, M, activation_fn=tf.nn.relu)

		out_policy = tf.contrib.layers.fully_connected(Z, num_actions, activation_fn=tf.nn.softmax)
		value      = tf.contrib.layers.fully_connected(Z, 1, activation_fn=lambda x: x)
		out_value  = tf.contrib.layers.flatten(value)

		self.predict_p = out_policy
		self.predict_v = out_value

		# calculate the loss
		selected_action_prob = tf.reduce_sum(out_policy * tf.one_hot(self.actions, num_actions), axis=1)
		log_prob = tf.log(selected_action_prob + 1e-10)
		advantage = self.returns - out_value

		loss_policy  = -log_prob * tf.stop_gradient(advantage) # maximize policy performance
		loss_value   = LOSS_VALUE * tf.square(advantage)       # minimize value error
		loss_entropy = LOSS_ENTROPY * tf.reduce_sum(out_policy * tf.log(out_policy + 1e-10), axis=1) # maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + loss_entropy)

		self.train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.99).minimize(loss_total)

		# set the session and initialize variables
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

		# avoid modifications
		self.default_graph = tf.get_default_graph()
		self.default_graph.finalize()

	def predict_policy(self, states):
		states = np.atleast_2d(states)
		return self.session.run(self.predict_p, feed_dict={self.states: states})

	def predict_value(self, states):
		states = np.atleast_2d(states)
		return self.session.run(self.predict_v, feed_dict={self.states: states})

	def predict(self, states):
		states = np.atleast_2d(states)
		outputs = [self.predict_p, self.predict_v]
		return self.session.run(outputs, feed_dict={self.states: states})

	def train_push(self, s, a, r, s2):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s2 is None:
				self.train_queue[3].append(TERMINAL_STATE)
				self.train_queue[4].append(0.)
			else:
				self.train_queue[3].append(s2)
				self.train_queue[4].append(1.)

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH_SZ:
			time.sleep(0) # yield
			return

		with self.lock_queue:
			# more thread could have passed without lock, we can't yield inside lock
			if len(self.train_queue[0]) < MIN_BATCH_SZ:
				return

			s, a, r, s2, s_mask = self.train_queue
			self.train_queue = [[], [], [], [], []]

		s, a, r, s2, s_mask = map(np.vstack, [s, a, r, s2, s_mask])

		if len(s) > MAX_BATCH_SZ:
			print('Optimizer alert! Minimizing batch of %d samples!' % len(s))

		v = self.predict_value(s2)
		r = r + GAMMA_N * v * s_mask # set v to 0 if s2 is terminal state

		self.session.run(self.train_op, feed_dict={self.states: s, self.actions: a, self.returns: r})


class Agent:

	def __init__(self, brain, num_actions, eps_start, eps_end, eps_steps):
		self.brain = brain
		self.num_actions = num_actions
		self.step_count = 0

		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps
		self.eps_delta = (eps_end - eps_start) / eps_steps

		self.memory = [] # used for n-step return
		self.R = 0.

	def add_brain(self, brain):
		self.brain = brain

	def get_epsilon(self):
		if self.step_count >= self.eps_steps:
			return self.eps_end
		else:
			return self.eps_start + self.step_count * self.eps_delta # linearly interpolates

	def select_action(self, s):
		eps = self.get_epsilon()
		self.step_count += 1

		if np.random.random() < eps:
			return np.random.choice(self.num_actions)
		else:
			policy = self.brain.predict_policy(s)[0]

			# a = np.argmax(policy)
			a = np.random.choice(self.num_actions, p=policy)

			return a
