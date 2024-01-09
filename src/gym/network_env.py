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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import sender_obs, sender

MAX_CWND = 5000
MIN_CWND = 4
BYTES_PER_PACKET = 1500
LINK_DELAY = 0.05
alpha = 0.95

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 12.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
COOKED_TRACE_FOLDER = './cooked_traces/'
sigma = 4


class Packet():
    def __init__(self, p_ID, spawn, sender):
        self.packet_ID = p_ID
        self.spawn_time = spawn
        self.sender = sender
        self.waiting_for_send = True


class Network:
    def __init__(
            self,
            loss_rate,
            senders: [],
            random_seed=RANDOM_SEED,
    ):

        np.random.seed(random_seed)
        self.task_list = [
            "bus.ljansbakken",
            "car.snaroya",
            "ferry.nesoddtangen",
            "metro.kalbakken",
            "norway_bus",
            "norway_car",
            "norway_metro",
            "norway_train",
            "norway_tram",
            "amazon",
            "yahoo",
            "facebook",
            "youtube",
        ]

        self.all_cooked_time, self.all_cooked_bw, self.all_cooked_filename = self.load_trace()

        assert len(self.all_cooked_time) == len(self.all_cooked_bw)

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.trace_idx = 92
        print("generate network env, use net idx: " + str(self.trace_idx))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.cur_time = 0.0
        self.senders = senders
        for sender in self.senders:
            sender.register_network(self)
        self.env_loss_rate = loss_rate
        self.env_pack_volume = 10 * len(self.senders)

    def get_random_gauss_val(self, K_RANGE):
        bottom, top = K_RANGE
        mean = (bottom + top) / 2
        val = random.gauss(mean, sigma)
        while val < bottom or val > top:
            val = random.gauss(mean, sigma)
        return val

    def get_cur_time(self):
        return self.cur_time

    def get_bandwidth(self):
        return self.cooked_bw[self.mahimahi_ptr]

    def reset(self, trace_idx=None):
        if trace_idx == None:
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
        else:
            self.trace_idx = trace_idx
        print("generate network env, use net idx: " + str(self.trace_idx), "net mahimahi steps: ",
              str(len(self.cooked_time)))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def stash_env_state(self):
        return [self.mahimahi_ptr, self.last_mahimahi_time]

    def unstash_env_state(self, mahimahi_state):
        self.mahimahi_ptr, self.last_mahimahi_time = mahimahi_state

    def increase_mahimahi(self):
        # if self.last_mahimahi_time<self.cooked_time[self.mahimahi_ptr]:
        #     return
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
        self.mahimahi_ptr += 1

        if self.mahimahi_ptr >= len(self.cooked_bw):
            # loop back in the beginning
            # note: trace file starts with time 0
            self.mahimahi_ptr = 1
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def run_for_dur(self, dur):
        for sender in self.senders:
            sender.reset_obs()

        start_time = self.cur_time
        end_time = self.cur_time + dur
        while self.cur_time < end_time:  # 平等的对待所有sender，每一轮每个sender都发一个包
            cnt = 0
            for sender in self.senders:
                while sender.can_send_packet():
                    p = Packet(0, self.cur_time, sender)
                    sender.send_packet(p)
                    cnt += 1

            if cnt == 0:  # 无包发送，手动过一段时间
                cur_batch_size = len(self.senders)*PACKET_SIZE
            else:
                cur_batch_size = cnt * PACKET_SIZE
            # delay = 0
            size = 0
            while self.cur_time < end_time:  # 将这一轮包发完
                throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if throughput * duration + size >= cur_batch_size:
                    fractional_time = (
                            (cur_batch_size - size)
                            / throughput
                    )
                    self.cur_time += fractional_time
                    size += throughput * fractional_time
                    self.last_mahimahi_time += fractional_time
                    assert self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr]
                    break
                size += throughput * duration
                self.cur_time += duration
                self.increase_mahimahi()

            if cnt>0:
                arrive_cnt = size // PACKET_SIZE + 1  # 到达量分配给每个sender
                while arrive_cnt > 0:
                    ok = False
                    for sender in self.senders:
                        if sender.arrive_cnt < len(sender.packetQueue):
                            ok = True
                            sender.arrive_cnt += 1
                            arrive_cnt -= 1
                    if not ok: break

            # 确认收到
            # while self.cur_time < end_time:
            for sender in self.senders:
                while sender.has_packet_arrive(self.cur_time):
                    if random.randint(0, 1000) / 1000 < self.env_loss_rate:  # 丢包了
                        sender.on_packet_lost()
                    else:
                        sender.on_packet_acked(self.cur_time)
                    sender.packetQueue = sender.packetQueue[1:]
                    sender.arrive_cnt -= 1

            # 只保留网络容量的包，其余的包全部丢失
            remain_packet_num = 0
            remain_sender = 0
            for sender in self.senders:
                if len(sender.packetQueue) > 0:
                    remain_sender += 1
                    remain_packet_num += len(sender.packetQueue)
            # 平等的丢包
            if remain_packet_num > self.env_pack_volume:
                avg_drop = (remain_packet_num - self.env_pack_volume) // remain_sender
                drop_delay = min(1.0, avg_drop * 0.1)
                self.cur_time += drop_delay
                self.cur_time = min(self.cur_time, end_time)

                for sender in self.senders:
                    if len(sender.packetQueue) > 0:
                        for i in range(0, avg_drop):
                            sender.packetQueue = sender.packetQueue[1:]
                            sender.on_packet_lost()

        sender_mis = []
        for sender in self.senders:
            sender_mis.append(sender.get_run_data())

        return sender_mis, self.cur_time - start_time

    def load_trace(self, cooked_trace_folder=COOKED_TRACE_FOLDER):
        cooked_files = os.listdir(cooked_trace_folder)
        all_cooked_time = []
        all_cooked_bw = []
        all_file_names = []
        for cooked_file in cooked_files:
            file_path = cooked_trace_folder + cooked_file
            cooked_time = []
            cooked_bw = []
            # print file_path
            with open(file_path, 'rb') as f:
                for line in f:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)

        return all_cooked_time, all_cooked_bw, all_file_names
