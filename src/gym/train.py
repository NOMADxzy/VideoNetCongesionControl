import gc, os
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv
from linearNet import train, buffer
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm

# env = gym.make('PccNs-v0')

SUMMARY_DIR = "./Results/sim"
add_str = "linearNet"
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
SENDERS_NUM = 100


env = SimulatedNetworkEnv(SENDERS_NUM)
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
total_total_reward = 0

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(STATE_DIM, ACTION_DIM, A_MAX, ram)
trainer.load_models(msg="cwnd", episode=20)


def clip_state(state: np.ndarray):
    state = state.transpose()
    rate = state[1][4:]
    other = np.asarray([state[2][-1], state[2][-1], state[3][-1], state[4][-1]])
    return np.hstack((rate, other))


def run_test(_ep):
    print("-- TEST START --")
    avg_senders_obs, each_sender_obs = env.reset()
    state = np.float32(clip_state(avg_senders_obs))

    avg_rewards = []
    avg_throughputs = []
    avg_latencies = []
    avg_losses = []
    for r in tqdm(range(MAX_EP_STEPS)):
        env.render()

        avg_actions, avg_action, strength = trainer.get_exploration_action(state)

        actions = []
        for i in range(0, len(env.senders)):
            actions.append(avg_action)
        new_obs, rewards, done, info, _ = env.step(actions)
        avg_rewards.append(np.mean(rewards))
        avg_throughputs.append(info["throughput"])
        avg_latencies.append(info["latency"])
        avg_losses.append(info["loss"])

        new_state = np.float32(clip_state(new_obs[0]))
        state = new_state

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
    avg_senders_obs, each_sender_obs = env.reset()
    state = np.float32(clip_state(avg_senders_obs))
    print('EPISODE :- ', _ep)

    for r in range(MAX_EP_STEPS):
        env.render()

        avg_actions, avg_action, strength = trainer.get_exploration_action(state)

        actions = []
        for i in range(0, len(env.senders)):
            actions.append(avg_action)
        new_obs, rewards, done, info,_ = env.step(actions)
        avg_reward = np.mean(rewards)
        print("avg_reward:" + str(avg_reward), "action:" + str(avg_action), info)

        new_state = np.float32(clip_state(new_obs[0]))
        ram.add(state, avg_actions, avg_reward, new_state)
        state = new_state

        trainer.optimize()
        if done:
            break

    run_test(_ep)
    gc.collect()
    if _ep % 10 == 0:
        trainer.save_models("cwnd", _ep)

writer.close()
print('Completed episodes')
