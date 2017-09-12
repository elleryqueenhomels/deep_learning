# Use a Double DQN with Prioritized Experience Replay
# to solve Atari Games


from __future__ import division, print_function

import gym
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop


# ------------------ Constants ------------------
ENV = 'Breakout-v0'

IMAGE_WIDTH  = 84
IMAGE_HEIGHT = 84
IMAGE_STACK  = 4

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00025


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


# ------------------ Classes ------------------
class Brain:

    def __init__(self, num_state, num_action):
        self.num_state  = num_state
        self.num_action = num_action

        self.model = self.create_model()
        self.target_model = self.create_model() # target network

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.num_action, activation=None))

        optim = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=optim)

        return model
