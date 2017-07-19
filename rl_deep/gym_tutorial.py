# A Tutorial to OpenAI Gym

import gym
# Wiki:
# https://github.com/openai/gym/wiki/CartPole-v0
# Environment page (full list):
# https://gym.openai.com/envs/CartPole-v0

# get the environment
env = gym.make('CartPole-v0')

# put ourselves in the start state
# it also returns the state
env.reset()
# Out[3]: array([-0.03082769, -0.01153319, -0.03060178, -0.0461242 ])

# what do the state variables mean?
# Num  Observation           Min       Max
#  0   Cart Position         -2.4      2.4
#  1   Cart Velocity         -Inf      Inf
#  2   Pole Angle            ~ -41.8°  ~ 41.8°
#  3   Pole Velocity At Tip  -Inf      Inf

box = env.observation_space

# In [5]: box
# Out[5]: Box(4,)

# In [6]: box.
# box.contains       box.high           box.sample         box.to_jsonable
# box.from_jsonable  box.low            box.shape

env.action_space

# In [7]: env.action_space
# Out[7]: Discrete(2)

# In [8]: env.action_space.
# env.action_space.contains       env.action_space.n              env.action_space.to_jsonable
# env.action_space.from_jsonable  env.action_space.sample


# do an action
# observation, reward, done, info = env.step(action)


# run through an episode
done = False
while not done:
	observation, reward, done, _ = env.step(env.action_space.sample())
	print('observation:', observation, 'reward:', reward, 'done:', done)

