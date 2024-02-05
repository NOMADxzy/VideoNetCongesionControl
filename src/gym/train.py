import gc, os
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv
from CNN import train, buffer
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm
import scipy.stats as stats

# env = gym.make('PccNs-v0')

summary_dir = os.path.join(*["Results", "sim"])
ts = time.strftime("%b%d-%H_%M_%S", time.gmtime())
log_file_path = os.path.join(*[summary_dir, ts])

model_dir = os.path.join(*[log_file_path, "checkpoints/"])

utils.check_folder(model_dir)
writer = SummaryWriter(log_file_path)
# log_file_name = log_file_path + "/log"

######Train teacher first
STATE_DIM = 8
ACTION_DIM = 9
A_MAX = 1
MEAN_ACT = ACTION_DIM//2

MAX_BUFFER = 100000
MAX_EPISODES = 1000  # 200
MAX_EP_STEPS = 100  # 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
SENDERS_NUM = 10


env = SimulatedNetworkEnv(SENDERS_NUM)
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
total_total_reward = 0

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(STATE_DIM, ACTION_DIM, ram, 1)
# trainer.load_models(msg="cwnd", episode=190)


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
    for _ in tqdm(range(MAX_EP_STEPS)):
        env.render()

        actions = []
        acts = []

        for i,state in enumerate(states):
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




test_record = True

for _ep in range(MAX_EPISODES):
    obs = env.reset()
    states = trans_state(obs)
    # state = np.float32(clip_state(avg_senders_obs))
    print('EPISODE :- ', _ep)
    info = {}

    avg_rewards = []
    avg_throughputs = []
    avg_latencies = []
    avg_losses = []
    avg_cwnds = []

    for r in range(MAX_EP_STEPS):
        env.render()

        actions = []
        acts = []
        use_expert = 0
        if "loss" in info :
            if info["loss"] > 0.1:
                use_expert = -1
            elif info["action"] < MEAN_ACT and info["cwnd"]<=0.01:
                use_expert = 1

        for state in states:
            action, act, strength = trainer.get_exploration_action(state)
            explore_act = act
            if use_expert == -1 and act >= MEAN_ACT:
                explore_act = 0
                action[0][explore_act],action[0][act] = action[0][act],action[0][explore_act]
            elif use_expert == 1 and act <= MEAN_ACT:
                explore_act = ACTION_DIM-1
                action[0][explore_act],action[0][act] = action[0][act],action[0][explore_act]
            actions.append(action)
            # exlore_act = np.random.normal(act, eps)
            acts.append(explore_act)
        new_obs, rewards, done, info, cwnds = env.step(acts)
        # time.sleep(0.5)
        avg_reward = np.mean(rewards)
        avg_cwnd = np.mean(cwnds)
        print("avg_reward:" + str(avg_reward), "cwnds:" + str(cwnds), info)
        avg_rewards.append(avg_reward)
        avg_throughputs.append(info["throughput"])
        avg_latencies.append(info["latency"])
        avg_losses.append(info["loss"])
        avg_cwnds.append(avg_cwnd)
        # if avg_reward<0:
        #     eps = min(eps+0.1, MAX_EPS)

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
        if r == MAX_EP_STEPS-1:
            trainer.noise_weight *= 0.965
            env.expert_prob *= 0.965
            if test_record:
                writer.add_scalar("avg_cwnds", np.mean(avg_cwnds), _ep)
                writer.add_scalar("avg_throughputs_mean", np.mean(avg_throughputs), _ep)
                writer.add_scalar("avg_latencies_mean", np.mean(avg_latencies), _ep)
                writer.add_scalar("avg_losses_mean", np.mean(avg_losses), _ep)
                writer.add_scalar("avg_rewards_mean", np.mean(avg_rewards), _ep)
            else:
                run_test(_ep)
        if done:
            break

    # run_test(_ep)
    gc.collect()
    if _ep % 10 == 0:
        trainer.save_models(model_dir,"cwnd", _ep)

writer.close()
print('Completed episodes')
