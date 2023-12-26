import gym
import network_sim
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv

# env = gym.make('PccNs-v0')

######Train teacher first
MAX_EPISODES = 10#200
MAX_EP_STEPS = 400#200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
senders_num = 5
env = SimulatedNetworkEnv(senders_num)
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
total_total_reward = 0

t1 = time.time()
for i in range(MAX_EPISODES):
    env.reset()
    total_reward = 0
    for j in range(MAX_EP_STEPS):
        actions = []
        for i in range(0,senders_num):
            action = env.action_space.sample()  # direct action for test
            actions.append(action)
        new_state, reward, done, info = env.step(actions)
        total_reward = np.sum(reward)
        print("sum_reward:" + str(total_reward))
        if done:
            break
    # total_total_reward += total_reward
print("average_total_reward test", float(total_total_reward) / MAX_EPISODES)
