import gc, os
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv
from CNN import train, buffer
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm

# env = gym.make('PccNs-v0')

SUMMARY_DIR = "./Results/sim"
add_str = "CNN"
summary_dir = os.path.join(*[SUMMARY_DIR, add_str])
summary_dir = os.path.join(*[summary_dir, "log"])
ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
log_file_path = os.path.join(*[summary_dir, ts])
utils.check_folder(log_file_path)
writer = SummaryWriter(log_file_path)
# log_file_name = log_file_path + "/log"

######Train teacher first
STATE_DIM = 8
ACTION_DIM = 9
A_MAX = 1

MAX_BUFFER = 100000
MAX_EPISODES = 1000  # 200
MAX_EP_STEPS = 1000  # 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
SENDERS_NUM = 1


env = SimulatedNetworkEnv(SENDERS_NUM)
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
total_total_reward = 0

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(STATE_DIM, ACTION_DIM, ram)
# trainer.load_models(msg="cwnd", episode=20)


def clip_state(state: np.ndarray):
    state = state.transpose()
    rate = state[1][4:]
    other = np.asarray([state[2][-1], state[2][-1], state[3][-1], state[4][-1]])
    return np.hstack((rate, other))


def trans_state(obs):
    avg_senders_obs, each_sender_obs = obs
    avg_senders_obs = avg_senders_obs.transpose()
    states = []
    avg_state = np.asarray([avg_senders_obs[1], avg_senders_obs[4]])
    for sender_obs in each_sender_obs:
        sender_obs = sender_obs.transpose()
        state = np.vstack((sender_obs, avg_state))
        states.append(state)
    return np.float32(states)


def run_test(_ep):
    print("-- TEST START --")
    obs = env.reset()
    states = trans_state(obs)

    avg_rewards = []
    avg_throughputs = []
    avg_latencies = []
    avg_losses = []
    for r in tqdm(range(MAX_EP_STEPS)):
        env.render()

        actions = []
        acts = []
        for state in states:
            action, act, strength = trainer.get_exploration_action(state)
            actions.append(action)
            acts.append(act)
        new_obs, rewards, done, info, _ = env.step(acts)

        avg_rewards.append(np.mean(rewards))
        avg_throughputs.append(info["throughput"])
        avg_latencies.append(info["latency"])
        avg_losses.append(info["loss"])

        states = trans_state(new_obs)

        if done:
            break

    avg_rewards_mean, avg_throughputs_mean, avg_latencies_mean, avg_losses_mean = \
        np.mean(avg_rewards), np.mean(avg_throughputs), np.mean(avg_latencies), np.mean(avg_losses)
    writer.add_scalar("avg_throughputs_mean", avg_throughputs_mean, _ep)
    writer.add_scalar("avg_latencies_mean", avg_latencies_mean, _ep)
    writer.add_scalar("avg_losses_mean", avg_losses_mean, _ep)
    writer.add_scalar("avg_rewards_mean", avg_rewards_mean, _ep)
    writer.flush()


for _ep in range(MAX_EPISODES):
    obs = env.reset()
    states = trans_state(obs)
    # state = np.float32(clip_state(avg_senders_obs))
    print('EPISODE :- ', _ep)

    for r in range(MAX_EP_STEPS):
        env.render()

        actions = []
        acts = []
        for state in states:
            action, act, strength = trainer.get_exploration_action(state)
            actions.append(action)
            acts.append(act)
        new_obs, rewards, done, info,cwnds = env.step(acts)
        avg_reward = np.mean(rewards)
        avg_cwnd = np.mean(cwnds)
        print("avg_reward:" + str(avg_reward), "cwnds:" + str(cwnds), info)

        new_states = trans_state(new_obs)
        anti_dup = set({})
        for i in range(0,len(env.senders)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            new_state = new_states[i]
            if reward not in anti_dup:
                anti_dup.add(reward)
                ram.add(state, action, reward, new_state)

        states = new_states
        trainer.optimize()
        if done:
            break

    run_test(_ep)
    gc.collect()
    if _ep % 10 == 0:
        trainer.save_models("cwnd", _ep)

writer.close()
print('Completed episodes')
