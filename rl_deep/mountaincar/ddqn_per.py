# Use Double DQN with PER to solve MountainCar
# PER: Prioritized Experience Replay (Proportional Prioritization)

import gym
import time
import numpy as np
import tensorflow as tf
from replay_memory import PrioritizedReplayMemory


# ------------------ Constants ------------------
ENV = 'MountainCar-v0'

HIDDEN_LAYER_SIZES = [72, 36, 6]

GAMMA = 0.99
BATCH_SZ = 32
MEMORY_CAPACITY = 100000
ALPHA = 0.6

LEARNING_RATE = 2e-3
BETA = None

EPS_START = 0.4
EPS_STOP  = 0.1
EPS_STEPS = 10000

TARGET_UPDATE_PERIOD = 1000

NUM_EPISODES = 500


# ------------------ Classes ------------------

# Create the Deep Q-Network
class Brain:

    def __init__(self, input_sz, output_sz, hidden_layer_sizes, gamma, learning_rate=LEARNING_RATE, beta=BETA):
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.gamma = gamma
        self.beta = beta
        self.beta_count = 0

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
        #   tf.reshape(Q_values, [-1]),
        #   indices
        # )

        if beta is None:
            cost = tf.nn.l2_loss(self.targets - selected_action_values)
        else:
            self.IS_weights = tf.placeholder(tf.float32, shape=(None, ), name='IS_weights')
            cost = tf.reduce_sum(tf.multipy(self.IS_weights, tf.square(self.targets - selected_action_values)))

        # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(cost)

        self.params = tf.trainable_variables() # collect the model params

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def set_session(self, session):
        self.session = session

    def predict(self, states):
        states = np.atleast_2d(states)
        return self.session.run(self.predict_op, feed_dict={self.states: states})

    def get_beta(self):
        return 1 - (1 - self.beta) * np.exp(-0.00001 * self.beta_count)

    def get_td_error(self, state, action, reward, next_state, done, target_network):
        # Double DQN, one sample
        max_action = np.argmax(self.predict(next_state)[0])
        next_output = target_network.predict(next_state)[0, max_action]
        done_mask = 0.0 if done else 1.0
        target = reward + self.gamma * next_output * done_mask

        prediction = self.predict(state)[0, action]
        td_error = abs(target - prediction)

        return td_error

    def optimize(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, target_network, priorities):
        # Double DQN, batch samples
        max_actions = np.argmax(self.predict(batch_next_states), axis=1)
        next_outputs = target_network.predict(batch_next_states)[np.arange(max_actions.shape[0]), max_actions]
        done_masks = np.invert(batch_dones).astype(np.float32)
        targets = batch_rewards + self.gamma * next_outputs * done_masks

        predictions = self.predict(batch_states)[np.arange(batch_actions.shape[0]), batch_actions]
        td_errors = np.abs(targets - predictions)

        if self.beta is None:
            feed_dict={self.states: batch_states, self.actions: batch_actions, self.targets: targets}
        else:
            # calculate Importance-Sampling Weights: W = (N * P)**(-beta) / max(W), P means priority
            beta = self.get_beta()
            self.beta_count += 1

            # N = self.memory.current_length()
            # IS_weights = (N * priorities) ** (-beta)
            IS_weights = priorities ** (-beta) # NOTE: N ** (-beta) will be divided out because of normalization
            IS_weights /= np.max(IS_weights) # normalize

            feed_dict={self.states: batch_states, self.actions: batch_actions, self.targets: targets, self.IS_weights: IS_weights}

        self.session.run(self.train_op, feed_dict=feed_dict)

        return td_errors

    def copy_from(self, other):
        ops = []
        for p, q in zip(self.params, other.params):
            op = p.assign(q.value())
            ops.append(op)
        self.session.run(ops)


class Agent:

    def __init__(self, brain, target_network, num_actions, batch_sz, memory_capacity, alpha, eps_start, eps_end, eps_steps, target_update_period):
        self.brain = brain
        self.target_network = target_network
        self.num_actions = num_actions
        self.batch_sz = batch_sz
        self.memory = PrioritizedReplayMemory(memory_capacity, alpha)
        self.target_update_period = target_update_period

        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps
        self.eps_delta = (eps_end - eps_start) / eps_steps
        self.step_count = 0
        self.target_update_count = 0

    def get_epsilon(self):
        if self.step_count >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + self.step_count * self.eps_delta # linearly interpolates

    def select_action(self, state):
        eps = self.get_epsilon()
        self.step_count += 1

        if np.random.random() < eps:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.brain.predict(state)[0])

    def train(self, state, action, reward, next_state, done):
        event = (state, action, reward, next_state, done)
        td_error = self.brain.get_td_error(state, action, reward, next_state, done, self.target_network)
        self.memory.push(event, td_error)

        if self.memory.current_length() > self.batch_sz:
            if self.target_update_count % self.target_update_period == 0:
                self.target_network.copy_from(self.brain)
            self.target_update_count += 1

            samples, indices, priorities = self.memory.sample(self.batch_sz)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = samples

            priorities = np.array(priorities) / self.memory.total_sum()

            td_errors = self.brain.optimize(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, self.target_network, priorities)
            self.memory.update(indices, td_errors)


def play_one_episode(env, agent, render=False):
    s = env.reset()

    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()

        a = agent.select_action(s)
        s2, r, done, info = env.step(a)

        agent.train(s, a, r, s2, done)

        s = s2
        total_reward += r

    return total_reward


# ------------------ Entry Point ------------------
if __name__ == '__main__':
    env = gym.make(ENV)

    num_state   = env.observation_space.shape[0]
    num_actions = env.action_space.n

    brain          = Brain(num_state, num_actions, HIDDEN_LAYER_SIZES, GAMMA, learning_rate=LEARNING_RATE, beta=BETA)
    target_network = Brain(num_state, num_actions, HIDDEN_LAYER_SIZES, GAMMA, learning_rate=LEARNING_RATE, beta=BETA)
    
    agent = Agent(brain, target_network, num_actions, BATCH_SZ, MEMORY_CAPACITY, ALPHA, EPS_START, EPS_STOP, EPS_STEPS, TARGET_UPDATE_PERIOD)

    total_rewards = np.zeros(NUM_EPISODES)
    for n in range(NUM_EPISODES):
        total_reward = play_one_episode(env, agent)
        total_rewards[n] = total_reward

        if n % 10 == 0:
            print('episode: %d, current reward: %s, last 100 episodes avg reward: %s' % (n, total_reward, total_rewards[max(0, n-99):(n+1)].mean()))

    print('avg reward for last 100 episodes: %s' % total_rewards[-100:].mean())

    # test
    agent.eps_start = 0.
    agent.eps_end   = 0.
    agent.eps_steps = 0.
    agent.eps_delta = 0.
    agent.step_count = 0
    while True:
        time.sleep(0.01)
        total_reward = play_one_episode(env, agent, render=True)
        print('Reward: %s' % total_reward)

