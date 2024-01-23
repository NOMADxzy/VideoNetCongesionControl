import gc, os
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv
from CNN import train, buffer
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm
import scipy.stats as stats

SUMMARY_DIR = "./Results/test"
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
MEAN_ACT = ACTION_DIM//2

MAX_BUFFER = 100000
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
SENDERS_NUM = 10


ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(STATE_DIM, ACTION_DIM, ram, 0)
trainer.load_models(msg="cwnd", episode=0, root='./Results/checkpoints/single/')

class Client:
    def __init__(self, trainer:train.Trainer, sender_num):
        self.agent = trainer
        self.sender_num = sender_num
        self.env = SimulatedNetworkEnv(sender_num)
        self.env.seed(1)
        self.selfish = True

    def run(self, step):
        obs = self.env.reset()
        self.env.net.set_net_ptr(148)

        states = self.trans_state(obs)
        info = {}
        cwnds = [1 for _ in range(0,self.sender_num)]
        best_avg_cwnd = 1
        for r in range(step):
            self.env.render()

            actions = []
            acts = []
            use_expert = 0
            if "loss" in info:
                if info["loss"] > 0.1:
                    use_expert = -1
                elif info["action"] < MEAN_ACT and info["cwnd"] <= 0.01:
                    use_expert = 1

            for i,state in states:
                if self.selfish:
                    if cwnds[i] <= best_avg_cwnd:
                        explore_act = ACTION_DIM-1
                    else:
                        explore_act = 0
                else:
                    action, act, strength = trainer.get_exploration_action(state)
                    explore_act = act
                    if use_expert == -1 and act >= MEAN_ACT:
                        explore_act = 0
                        action[0][explore_act], action[0][act] = action[0][act], action[0][explore_act]
                    elif use_expert == 1 and act <= MEAN_ACT:
                        explore_act = ACTION_DIM - 1
                        action[0][explore_act], action[0][act] = action[0][act], action[0][explore_act]
                    actions.append(action)

                acts.append(explore_act)

            new_obs, rewards, done, info, cwnds = self.env.step(acts)

            writer.add_scalar("avg_reward", np.mean(rewards), r)
            writer.add_scalar("avg_throughputs_mean", info["throughput"], r)
            writer.add_scalar("avg_latencies_mean", info["latency"], r)
            writer.add_scalar("avg_losses_mean", info["loss"], r)
            writer.add_scalar("avg_cwnd_mean", info["cwnd"], r)
            writer.add_scalar("avg_action_mean", info["action"], r)

            new_states = self.trans_state(new_obs)

            states = new_states
            if done:
                break

        gc.collect()

    def trans_state(self,obs):
        avg_senders_obs, each_sender_obs = obs
        avg_senders_obs = avg_senders_obs.transpose()
        states = []
        avg_state = np.asarray([avg_senders_obs[1], avg_senders_obs[4]])
        for sender_obs in each_sender_obs:
            sender_obs = sender_obs.transpose()
            state = np.vstack((sender_obs, avg_state))
            states.append(state)
        return np.float32(states)


if __name__ == "__main__":
    c = Client(trainer, 10)
    c.run(100)
