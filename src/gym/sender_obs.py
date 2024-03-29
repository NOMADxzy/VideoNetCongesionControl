# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

# The monitor interval class used to pass data from the PCC subsystem to
# the machine learning module.
#
class SenderMonitorInterval():
    def __init__(self,
                 sender_id,
                 bytes_sent=0.0,
                 bytes_acked=0.0,
                 bytes_lost=0.0,
                 send_start=0.0,
                 send_end=0.0,
                 recv_start=0.0,
                 recv_end=0.0,
                 rtt_samples=[],
                 packet_size=1500,
                 link0_bw=0,
                 link1_bw=0,
                 utilization=[],
                 utilization0=[],
                 aoi=0,
                 cwnd=0.0,
                 last_cwnd_action=0.0):
        self.features = {}
        self.sender_id = sender_id
        self.bytes_acked = bytes_acked
        self.bytes_sent = bytes_sent
        self.bytes_lost = bytes_lost
        self.send_start = send_start
        self.send_end = send_end
        self.recv_start = recv_start
        self.recv_end = recv_end
        self.rtt_samples = rtt_samples
        self.packet_size = packet_size
        self.link0_bw = link0_bw
        self.link1_bw = link1_bw
        self.utilization = utilization
        self.utilization0 = utilization0
        self.aoi=aoi
        self.cwnd=cwnd
        self.last_cwnd_action=last_cwnd_action
    def get_wrong(self, feature):
        return self.sender_id
    def get(self, feature):
        #print(feature)
        #print(self.features.keys())
        if feature in self.features.keys():
            return self.features[feature]
        else:
            result = SenderMonitorIntervalMetric.eval_by_name(feature, self)
            self.features[feature] = result
            return result

    # Convert the observation parts of the monitor interval into a numpy array
    def as_array(self, features):
        #print(self.get("send rate"))
        #print(SenderMonitorIntervalMetric.get_by_name("send rate").scale)
        return [self.get(f) / SenderMonitorIntervalMetric.get_by_name(f).scale for f in features]

class SenderHistory():
    def __init__(self, length, features, sender_id):
        self.features = features
        self.values = []
        self.sender_id = sender_id
        for i in range(0, length):
            self.values.append(SenderMonitorInterval(self.sender_id))

    def step(self, new_mi):
        self.values.pop(0)
        self.values.append(new_mi)

    def as_array(self):
        arrays = []

        for mi in self.values:
            #print (arrays)
            #print(mi.as_array("send rate"))
            arrays.append(mi.as_array(self.features))
        # arrays = np.array(arrays).flatten()
        return np.asarray(arrays) # 历史长度 * 特征维度

class SenderMonitorIntervalMetric():
    _all_metrics = {}

    def __init__(self, name, func, min_val, max_val, scale=1.0):
        self.name = name
        self.func = func
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale
        SenderMonitorIntervalMetric._all_metrics[name] = self

    def eval(self, mi):
        return self.func(mi)

    def eval_by_name(name, mi):
        return SenderMonitorIntervalMetric._all_metrics[name].eval(mi)

    def get_by_name(name):
        return SenderMonitorIntervalMetric._all_metrics[name]

def get_min_obs_vector(feature_names):
    #print("Getting min obs for %s" % feature_names)
    result = []
    for feature_name in feature_names:
        feature = SenderMonitorIntervalMetric.get_by_name(feature_name)
        result.append(feature.min_val)
    return np.array(result) 

def get_max_obs_vector(feature_names):
    result = []
    for feature_name in feature_names:
        feature = SenderMonitorIntervalMetric.get_by_name(feature_name)
        result.append(feature.max_val)
    return np.array(result) 

def _mi_metric_recv_rate(mi):
    dur = mi.get("recv dur")
    if dur > 0.0:
        #print(mi.bytes_acked," ",mi.packet_size," ",dur)
        return 8.0 * mi.bytes_acked / dur
    return 0.0

def _mi_metric_recv_dur(mi):
    return mi.recv_end - mi.recv_start

def _mi_metric_avg_latency(mi):
    
    #print(mi.link0_bw)
    #print(len(mi.rtt_samples))
    #print(mi.utilization)
    
    if len(mi.rtt_samples) > 0:
        #print(np.array(mi.rtt_samples))
        return np.mean(mi.rtt_samples)
    return 0.0

def _mi_metric_send_rate(mi):
    dur = mi.get("send dur")
    if dur > 0.0:
        return 8.0 * mi.bytes_sent / dur
    return 0.0

def _mi_metric_send_dur(mi):
    return mi.send_end - mi.send_start

def _mi_metric_loss_ratio(mi):
    if mi.bytes_lost + mi.bytes_acked > 0:
        return mi.bytes_lost / (mi.bytes_lost + mi.bytes_acked)
    return 0.0

def _mi_metric_latency_increase(mi):
    half = int(len(mi.rtt_samples) / 2)
    if half >= 1:
        return np.mean(mi.rtt_samples[half:]) - np.mean(mi.rtt_samples[:half])
    return 0.0

def _mi_metric_ack_latency_inflation(mi):
    dur = mi.get("recv dur")
    latency_increase = mi.get("latency increase")
    if dur > 0.0:
        return latency_increase / dur
    return 0.0

def _mi_metric_sent_latency_inflation(mi):
    dur = mi.get("send dur")
    latency_increase = mi.get("latency increase")
    if dur > 0.0:
        return latency_increase / dur
    return 0.0

_conn_min_latencies = {}
def _mi_metric_conn_min_latency(mi):
    latency = mi.get("avg latency")
    if mi.sender_id in _conn_min_latencies.keys():
        prev_min = _conn_min_latencies[mi.sender_id]
        if latency == 0.0:
            return prev_min
        else:
            if latency < prev_min:
                _conn_min_latencies[mi.sender_id] = latency
                return latency
            else:
                return prev_min
    else:
        if latency > 0.0:
            _conn_min_latencies[mi.sender_id] = latency
            return latency
        else:
            return 0.0
        
    
def _mi_metric_send_ratio(mi):
    thpt = mi.get("recv rate")
    send_rate = mi.get("send rate")
    if (thpt > 0.0) and (send_rate < 1000.0 * thpt):
        return send_rate / thpt
    return 1.0

def _mi_metric_latency_ratio(mi):
    min_lat = mi.get("conn min latency")
    cur_lat = mi.get("avg latency")
    if min_lat > 0.0:
        return cur_lat / min_lat
    return 1.0
def _mi_metric_utilization(mi):
    #print("test".x)
    if len(mi.utilization) > 0:
        #print(np.array(mi.rtt_samples))
        return np.mean(mi.utilization)
    return 0.0
def _mi_metric_utilization0(mi):
    #print("test".x)
    if len(mi.utilization0) > 0:
        #print(np.array(mi.rtt_samples))
        return np.mean(mi.utilization0)
    return 0.0
def _mi_metric_bw1(mi):
    return mi.link0_bw
def _mi_metric_bw2(mi):
    #print("test")
    return mi.link1_bw
def _mi_metric_aoi(mi):
    return mi.aoi

def _mi_metric_last_cwnd_action(mi):
    return mi.last_cwnd_action

def _mi_metric_cwnd(mi):
    return mi.cwnd

SENDER_MI_METRICS = [
    SenderMonitorIntervalMetric("send rate", _mi_metric_send_rate, 1e5, 1e8, 1e5),
    SenderMonitorIntervalMetric("recv rate", _mi_metric_recv_rate, 1e5, 1e8, 1e5),
    SenderMonitorIntervalMetric("recv dur", _mi_metric_recv_dur, 0.0, 100.0),
    SenderMonitorIntervalMetric("send dur", _mi_metric_send_dur, 0.0, 100.0),
    SenderMonitorIntervalMetric("avg latency", _mi_metric_avg_latency, 0.0, 100.0),
    SenderMonitorIntervalMetric("loss ratio", _mi_metric_loss_ratio, 0.0, 1.0),
    SenderMonitorIntervalMetric("ack latency inflation", _mi_metric_ack_latency_inflation, 0, 10.0),
    SenderMonitorIntervalMetric("sent latency inflation", _mi_metric_sent_latency_inflation, 0, 10.0),
    SenderMonitorIntervalMetric("conn min latency", _mi_metric_conn_min_latency, 0.0, 100.0),
    SenderMonitorIntervalMetric("latency increase", _mi_metric_latency_increase, 0.0, 100.0),
    SenderMonitorIntervalMetric("latency ratio", _mi_metric_latency_ratio, 1.0, 10000.0),
    SenderMonitorIntervalMetric("send ratio", _mi_metric_send_ratio, 0.0, 1000.0),
    SenderMonitorIntervalMetric("avg utilization", _mi_metric_utilization, 0.0, 1.0,1e-2),
    SenderMonitorIntervalMetric("avg utilization0", _mi_metric_utilization0, 0.0, 1.0,1e-2),
    SenderMonitorIntervalMetric("Link1 bandwidth", _mi_metric_bw1, 0.0, 1.0),
    SenderMonitorIntervalMetric("Link2 bandwidth", _mi_metric_bw2, 0.0, 1.0),
    SenderMonitorIntervalMetric("aoi", _mi_metric_aoi, 0.0, 200.0),
    SenderMonitorIntervalMetric("last_cwnd_action", _mi_metric_last_cwnd_action, -1.0, 1.0),
    SenderMonitorIntervalMetric("cwnd", _mi_metric_cwnd, 0, 1.0)
]


