# Use a Double DQN with Prioritized Experience Replay
# to solve Atari Games


from __future__ import division, print_function

import gym
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Permute, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop

from scipy.misc import imresize
from replay_memory import PrioritizedReplayMemory


# ------------------ Constants ------------------
ENV = 'Breakout-v0'

IMAGE_WIDTH  = 84
IMAGE_HEIGHT = 84
IMAGE_STACK  = 4

HUBER_LOSS_DELTA = 2.0

LEARNING_RATE = 0.00025
BATCH_SZ = 32

MEMORY_CAPACITY = 200000
ALPHA = 0.6

GAMMA = 0.99

MAX_EPSILON = 1.0
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000 # at this step epsilon will be 0.1
LAMBDA = -np.log(0.01) / EXPLORATION_STOP # speed of decay

UPDATE_TARGET_FREQUENCY = 10000


# ------------------ Utilities ------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2_loss = 0.5 * K.square(err)
    L1_loss = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2_loss, L1_loss)

    return K.mean(loss)


def process_image(img):
    rgb = imresize(img, size=(IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b # the effective luminance of a pixel

    out = gray.astype('float32') / 128 - 1 # normalize
    return out


def update_state(state, image):
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    state = np.append(state[1:], image, axis=0)
    return state


# ------------------ Classes ------------------
class Brain:

    def __init__(self, num_state, num_action):
        self.num_state  = num_state
        self.num_action = num_action

        self.model = self.create_model()
        self.target_model = self.create_model() # target network
        # self.update_target_model()

    def create_model(self):
        model = Sequential()

        model.add(Permute((2, 3, 1), input_shape=self.num_state))
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.num_action, activation=None))

        optim = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=optim)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, states, target=False):
        if target:
            return self.target_model.predict(states)
        else:
            return self.model.predict(states)

    def predict_one(self, state, target=False):
        states = state.reshape(1, IMAGE_STACK, IMAGE_HEIGHT, IMAGE_WIDTH)
        return self.predict(states, target)[0]

    def get_td_error(self, state, action, reward, next_state, done):
        best_action = np.argmax(self.predict_one(next_state))
        next_return = self.predict_one(next_state, target=True)[best_action]
        done_mask   = 0.0 if done else 1.0
        target      = reward + GAMMA * next_return * done_mask

        prediction  = self.predict_one(state)[action]
        td_error    = abs(target - prediction)

        return td_error

    def train(self, states, actions, rewards, next_states, dones, epochs=1, verbose=0):
        batch_range  = np.arange(len(actions))

        best_actions = np.argmax(self.predict(next_states), axis=1)
        next_returns = self.predict(next_states, target=True)[batch_range, best_actions]
        done_masks   = np.invert(dones).astype(np.float32)
        targets      = rewards + GAMMA * next_returns * done_masks

        outputs      = self.predict(states)
        predictions  = outputs[batch_range, actions]
        td_errors    = np.abs(targets - predictions)

        x = states
        y = outputs
        y[batch_range, actions] = targets

        self.model.fit(x, y, batch_size=BATCH_SZ, epochs=epochs, verbose=verbose)

        return td_errors


class Agent:

    def __init__(self, num_state, num_action):
        self.num_state  = num_state
        self.num_action = num_action

        self.brain  = Brain(num_state, num_action)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY, alpha=ALPHA)

        self.steps = 0
        self.epsilon = MAX_EPSILON

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_action)
        else:
            return np.argmax(self.brain.predict_one(state))

    def train(self, state, action, reward, next_state, done):
        event = (state, action, reward, next_state, done)
        td_error = self.brain.get_td_error(state, action, reward, next_state, done)
        self.memory.push(event, td_error)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target_model()

        # slowly decrease epsilon based on experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)

        # train the brain if the memory has enough experience
        if self.memory.current_length() > BATCH_SZ:
            samples, indices, priorities = self.memory.sample(BATCH_SZ)
            states, actions, rewards, next_states, dones = samples

            td_errors = self.brain.train(states, actions, rewards, next_states, dones)
            self.memory.update(indices, td_errors)


def play_one_episode(env, agent, render=False):
    # still under-development
    pass

