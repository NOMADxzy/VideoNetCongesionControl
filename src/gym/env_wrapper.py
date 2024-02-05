import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
import sys
import inspect
import math
import pdb
from sender import MIN_CWND,MAX_CWND

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import sender_obs, sender, network_env

MAX_STEPS = 10000
ACTION_DIM = 9
LATENCY_PENALTY = 1
LOSS_PENALTY = 4
THROUGHPUT_SCALAR = 1e5
BIT_IN_BYTES = 8
TEST_TRACE_IDX = 1
MAX_BURST_PACKETS = 2
BYTES_PER_PACKET = 1500


class SimulatedNetworkEnv(gym.Env):
    def __init__(self,
                 sender_num=5,
                 history_len=8,
                 features="send rate,"
                          "recv rate,"
                          "avg latency,"
                          "latency increase,"
                          "loss ratio,"
                          "cwnd"):
        self.features = features.split(",")
        self.featureIdx = {}
        for i,feature in enumerate(self.features):
            self.featureIdx[feature] = i
        self.viewer = None
        self.rand = None
        self.reward_list = []
        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.001)
        self.history_len = history_len
        # print("History length: %d" % history_len)
        # print("Features: %s" % str(self.features))
        self.senders = None
        self.net:network_env.Network
        self.create_network_and_senders(self.features, sender_num)
        self.run_dur = 0.2
        self.steps_taken = 0
        self.max_steps = MAX_STEPS

        self.last_thpt = None
        self.last_rate = None
        self.last_time = 0

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.action_mean = ACTION_DIM // 2

        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.step_reward = 0.0

        self.event_record = {"Events": []}
        self.episodes_run = -1
        self.expert_prob = 1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        each_sender_obs = []
        avg_senders_obs = np.zeros((self.history_len, len(self.features)))
        for sender in self.senders:
            obs = sender.get_obs()
            each_sender_obs.append(obs)
            avg_senders_obs += obs
        avg_senders_obs /= len(self.senders)
        return avg_senders_obs, each_sender_obs

    def step(self, actions):

        assert len(actions) == len(self.senders)
        for i, sender in enumerate(self.senders):
            delta = (actions[i] - self.action_mean) / self.action_mean
            sender.apply_cwnd_delta(delta)

        sender_mis, dur, best_avg_cwnd = self.net.run(self.run_dur)

        # for sender in self.senders:
        #     sender.record_run()
        self.steps_taken += 1
        avg_sender_obs, each_sender_obs = self._get_all_sender_obs()

        rewards,throughputs,latencies,losses,cwnds = [], [], [], [], []
        for i,sender_mi in enumerate(sender_mis):
            # sender = self.senders[i]
            throughput = sender_mi.get("recv rate")
            latency = sender_mi.get("avg latency")
            loss = sender_mi.get("loss ratio")
            cwnd = sender_mi.get("cwnd")

            throughputs.append(throughput)
            latencies.append(latency)
            losses.append(loss)
            cwnds.append(cwnd)

            reward = 2*(throughput / (BIT_IN_BYTES * THROUGHPUT_SCALAR) - LATENCY_PENALTY * latency - LOSS_PENALTY * loss)
            if loss>0.01: # 存在丢包
                # 按动作大小调整惩罚
                if actions[i]==0:
                    reward = 1
                else:
                    delta = 1 + (actions[i]-1 - self.action_mean) / self.action_mean
                    reward *= delta
            elif best_avg_cwnd > cwnd and cwnd<MIN_CWND*10*2/MAX_CWND:
                reward = 0
            # if (sender.cwnd - MAX_BURST_PACKETS) * BYTES_PER_PACKET < sender.bytes_in_flight:
            #     reward -= 10
            # if cwnd == 0.008 and np.random.randint(0,1000)/1000<self.expert_prob:
            #     reward = -1
            rewards.append(reward)
        avg_reward = sum(rewards)/len(rewards)
        self.reward_list.append(avg_reward)

        # "send rate,avg latency,latency increase,loss ratio,last_cwnd_action"
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Bandwidth"] = self.net.get_bandwidth()
        event["AVG Reward"] = avg_reward
        event["Send Rate"] = avg_sender_obs[-1][self.featureIdx["send rate"]]
        event["Avg Latency"] = avg_sender_obs[-1][self.featureIdx["avg latency"]]
        event["Latency Increase"] = avg_sender_obs[-1][self.featureIdx["latency increase"]]
        event["loss ratio"] = avg_sender_obs[-1][self.featureIdx["loss ratio"]]
        event["cwnd"] = avg_sender_obs[-1][self.featureIdx["cwnd"]]

        # event["Cwnd"] = sender_mi.cwnd
        # event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)

        """
        if self.steps_taken%10==0:
            print("steps_taken",self.steps_taken)
        """
        return [avg_sender_obs, each_sender_obs], rewards, (self.steps_taken >= self.max_steps), \
            {"step": self.steps_taken, "action": np.mean(actions), "throughput":np.mean(throughputs),
             "latency":np.mean(latencies), "loss":np.mean(losses), "cwnd": np.mean(cwnds), "best_avg_cwnd":best_avg_cwnd}, cwnds  # ,123

    def print_debug(self):
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_network_and_senders(self, features, sender_num):
        lat = random.uniform(self.min_lat, self.max_lat)
        loss = random.uniform(self.min_loss, self.max_loss)
        self.senders = [
            sender.Sender(i, features=features, history_len=self.history_len) for i in range(0, sender_num)]
        self.net = network_env.Network(loss, self.senders)
        self.run_dur = 3 * lat

    def reset(self):
        self.steps_taken = 0
        # self.net.reset(93)
        self.net.reset()

        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            # self.dump_events_to_file("828ver0aoidur05_log_run_%d.json" % self.episodes_run, 0)
            pass
        self.event_record = {"Events": []}
        # self.net.run(self.run_dur)
        # self.net.run_for_dur(self.run_dur)
        # self.dump_events_to_file("828ver3dur05aoi_log_reward.json", 1)
        for sender in self.senders:
            sender.reset()
        avg_senders_obs, each_sender_obs = self._get_all_sender_obs()
        return [avg_senders_obs, each_sender_obs]

    def render(self, mode='human'):
        pass

    def dump_events_to_file(self, filename, index):
        if index == 0:
            with open(filename, 'w') as f:
                json.dump(self.event_record, f, indent=5)
        else:
            with open(filename, 'w') as f:
                json.dump(self.reward_list, f, indent=5)


# register(id='PccNs-v0', entry_point='network_sim:SimulatedNetworkEnv')