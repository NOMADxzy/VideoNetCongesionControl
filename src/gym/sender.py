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
#import tensorflow as tf
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import sender_obs

MAX_CWND = 5000.0
MIN_CWND = 4.0
BYTES_PER_PACKET = 1500

alpha=0.95
RANDOM_SEED = 42
LINK_DELAY = 0.1


class Sender():
    # [Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
    def __init__(self, sender_id, features, history_len, cwnd=MIN_CWND):
        self.id = sender_id
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.net = None

        self.obs_start_time = 0
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

        self.packetQueue = []
        self.last_cwnd_action = 0.0
        self.arrive_cnt = 0


    def get_bytes_in_flight(self):
        return self.bytes_in_flight

    def apply_cwnd_delta(self, delta):
        self.last_cwnd_action = delta
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))
        # self.cwnd = 1

    def send_packet(self, p):
        self.on_packet_sent()
        self.packetQueue.append(p)

    def can_send_packet(self):
        return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd

    def find_first_need_send(self):
        l,r = 0,len(self.packetQueue)-1
        if self.packetQueue[l].waiting_for_send:
            return l
        while l<r:
            mid = (l+r) // 2
            if self.packetQueue[mid].waiting_for_send:
                r = mid
            else:
                l = mid+1
        return l

    def has_packet_arrive(self, now):
        return len(self.packetQueue)>0 and self.arrive_cnt>0 and (now - self.packetQueue[0].spawn_time) > LINK_DELAY
            # and not self.packetQueue[0].waiting_for_send

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, now):
        self.acked += 1
        rtt = now - self.packetQueue[0].spawn_time + LINK_DELAY
        self.rtt_samples.append(rtt)

        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET


    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        # print(smi)
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()

        obs_dur = obs_end_time - self.obs_start_time
        # print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        # print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        # print("self.rate = %f" % self.rate)
        # print(self.aoi_cur_time)
        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET,
            cwnd=self.cwnd*10/MAX_CWND,
            last_cwnd_action=self.last_cwnd_action
        )

    ###############################################ACP#################
    def reset_obs(self):
        self.obs_start_time = self.net.get_cur_time()
        self.sent = self.bytes_in_flight / BYTES_PER_PACKET
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []


    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        # print("Resetting sender!")
        self.cwnd = MIN_CWND
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)