import gc, os
import numpy as np
import time
from env_wrapper import SimulatedNetworkEnv
from CNN import train, buffer
from torch.utils.tensorboard import SummaryWriter
import utils
from enum import Enum
from tqdm import tqdm
import scipy.stats as stats

SUMMARY_DIR = "./Results/test"
summary_dir = os.path.join(*["Results", "test"])
ts = time.strftime("%b%d-%H_%M_%S", time.gmtime())

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

class Policy(Enum):
    CNN = 1
    MFG = 2
    SELFISH = 3
    RENO = 4
    MFG_FIX = 5

class Client:
    def __init__(self, trainer:train.Trainer, sender_num, policy=Policy.SELFISH):
        self.agent = trainer
        self.sender_num = sender_num
        self.env = SimulatedNetworkEnv(sender_num)
        self.env.seed(1)
        self.policy = policy
        self.reno_thres = 16
        self.action_mean = ACTION_DIM // 2

    def generateWriter(self):
        ts = (str(self.policy)[7:] + "_sender" + str(self.sender_num) + "_net" +
              str(self.env.net.trace_idx) + "_trace") + str(self.env.net.mahimahi_ptr)
        log_file_path = os.path.join(*[summary_dir, ts])
        utils.check_folder(log_file_path)
        writer = SummaryWriter(log_file_path)
        return writer

    def applyAct(self, cwnd, act):
        delta = (act - self.action_mean) / self.action_mean
        return cwnd * (1 + delta)

    def run(self, step):
        obs = self.env.reset()
        self.env.net.set_net_ptr(14)
        # self.env.net.set_net_ptr(90)
        writer = self.generateWriter()

        states = self.trans_state(obs)
        info = {}
        cwnds = [1 for _ in range(0, self.sender_num)]
        best_avg_cwnd = 0
        for r in range(step):
            self.env.render()

            acts = []
            use_expert = 0
            if "loss" in info:
                if info["loss"] > 0.1:
                    use_expert = -1
                elif info["action"] < MEAN_ACT and info["cwnd"] <= 0.01:
                    use_expert = 1

            for i,state in enumerate(states):
                # MFG策略
                if self.policy == Policy.MFG:
                    if cwnds[i] <= best_avg_cwnd:
                        explore_act = ACTION_DIM - 1
                        while self.applyAct(cwnds[i], explore_act) > best_avg_cwnd:
                            explore_act -= 1
                        explore_act += 1
                    else:
                        explore_act = 0
                        while self.applyAct(cwnds[i], explore_act) < best_avg_cwnd:
                            explore_act += 1
                        explore_act -= 1
                # 自私策略
                elif self.policy == Policy.SELFISH:
                    if "loss" in info and info["loss"] > 0.1:
                        explore_act = 1
                    else:
                        explore_act = ACTION_DIM-2
                # CNN模型
                elif self.policy == Policy.CNN:
                    action, act, strength = self.agent.get_exploration_action(state)
                    explore_act = act
                    if use_expert == -1 and act >= MEAN_ACT:
                        explore_act = 0
                        action[0][explore_act], action[0][act] = action[0][act], action[0][explore_act]
                    elif use_expert == 1 and act <= MEAN_ACT:
                        explore_act = ACTION_DIM - 1
                        action[0][explore_act], action[0][act] = action[0][act], action[0][explore_act]
                # reno策略
                elif self.policy == Policy.RENO:
                    if "loss" in info and info["loss"] >= 0.1:
                        self.reno_thres /= 2
                        self.env.senders[i].set_cwnd(self.reno_thres)
                    else:
                        cur_cwnd = cwnds[i] * 5000 / 10
                        if cur_cwnd < self.reno_thres:
                            self.env.senders[i].set_cwnd(cur_cwnd * 2)
                        else:
                            self.env.senders[i].set_cwnd(cur_cwnd + 1)
                    explore_act = MEAN_ACT
                elif self.policy == Policy.MFG_FIX:
                    if cwnds[i] <= best_avg_cwnd:
                        explore_act = ACTION_DIM - 4
                    else:
                        explore_act = 3
                else:
                    raise ValueError

                acts.append(explore_act)

            new_obs, rewards, done, info, cwnds = self.env.step(acts)
            best_avg_cwnd = info["best_avg_cwnd"]

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

def get_agent():
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    trainer = train.Trainer(STATE_DIM, ACTION_DIM, ram, 0)
    # trainer.load_models(msg="cwnd", episode=150, root='.\\Results\\cktpts\\insupress_10\\')
    trainer.load_models(msg="cwnd", episode=980, root='.\\Results\\cktpts\\supress_10\\')
    return trainer

if __name__ == "__main__":
    policys = [Policy.CNN, Policy.RENO, Policy.MFG_FIX, Policy.MFG]
    for p in policys:
        c = Client(get_agent(), 1, p)
        c.run(100)

