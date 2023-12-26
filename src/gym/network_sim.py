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
import random
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
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default

MAX_CWND = 5000
MIN_CWND = 4

REWARD_CONTROL = 3
sender2_TCP = False
MAX_RATE = 1000
MIN_RATE = 1
packet_return_size = 1000
REWARD_SCALE = 0.1
Change_rate_step = 200
Sender0_mode = 0 #0:RL 1:TCP 2:ACP
MAX_STEPS = 400
step_control = 0
EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'
EVENT_TYPE_RETURN = 'R'
EVENT_TYPE_FIN = 'F'
BYTES_PER_PACKET = 1500
LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0
USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.001
USE_CWND = False
alpha=0.95
ACP_step_size=1

def debug_pack(pack):
    print(pack.packet_ID)
    print(pack.content)
    print(pack.spawn_time)
    print(pack.now_link)
    print(pack.event_type)
    print(pack.path)
    print(pack.path_num)
    print(pack.home_path)
class Packet():
    def __init__(self, p_ID,spawn,path_num,path,con=[]):
        self.packet_ID=p_ID
        self.content=con
        self.spawn_time=spawn
        self.now_link=[0,0]
        self.event_type=EVENT_TYPE_SEND
        self.path=path
        self.path_num=0
        self.home_path=[]
        self.link0_bw=0
        self.link1_bw=0
        self.utilization=0
        self.utilization0=0
    def reset(self):
        self.packet_ID=0
        self.content=[]
        self.spawn_time=0
        self.now_link=[0,0]
        self.event_type=EVENT_TYPE_SEND
        self.path=[]
        self.home_path=[]
    
class Link():
    def __init__(self, link_id , bandwidth, delay, queue_size, loss_rate,neibor,start_point,end_point):
        self.bw = float(bandwidth)
        self.dl = delay
        self.neibor = neibor
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.return_queue_delay = 0.0
        self.link_id=link_id
        self.queue_delay_update_time = 0.0
        self.return_queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.end_point=end_point
        self.start_point=start_point
    def get_cur_queue_delay(self, event_time,event_type):
        if event_type == EVENT_TYPE_RETURN:
            return max(0.0, self.return_queue_delay - (event_time - self.return_queue_delay_update_time))            
        else:
            return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time,event_type):
        if event_type==EVENT_TYPE_RETURN :
            return 0.001*self.dl + self.get_cur_queue_delay(event_time,event_type)
        else:
            return self.dl + self.get_cur_queue_delay(event_time,event_type)
    def packet_enters_link(self, event_time,event_type):

        
        
        if event_type==EVENT_TYPE_RETURN:
            self.return_queue_delay= self.get_cur_queue_delay(event_time,event_type)
            self.return_queue_delay_update_time = event_time
            extra_delay = 1/packet_return_size / self.bw
            self.return_queue_delay += extra_delay
            return True
        else:
            if (random.random() < self.lr):
                return False
            self.queue_delay = self.get_cur_queue_delay(event_time,event_type)
            self.queue_delay_update_time = event_time
            extra_delay = 1.0 / self.bw
            #print(self.link_id,"Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
            if extra_delay + self.queue_delay > self.max_queue_delay:
                #print("\tDrop!")
                return False
            self.queue_delay += extra_delay
            #print("\tNew delay = %f" % self.queue_delay)
            return True
    def state_change(self):
        if (random.random() < (1.0/3)):
            self.bw_change()
        elif  (random.random() < (2.0/3)):
            self.dl_change()
        else:
            self.lr_change()
    def bw_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.bw=self.bw * (1.0 + delta)
        else:
            self.bw=self.bw / (1.0 - delta)
        if self.bw>self.max_bw:
            self.bw=self.max_bw
        elif self.bw<self.min_bw:
            self.bw=self.min_bw
    def dl_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.dl=self.dl * (1.0 + delta)
        else:
            self.dl=self.dl / (1.0 - delta)
        if self.dl>self.max_lat:
            self.dl=self.max_lat
        elif self.dl<self.min_lat:
            self.dl=self.min_lat
    def lr_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.lr=self.lr * (1.0 + delta)
        else:
            self.lr=self.lr / (1.0 - delta)
        if self.lr>self.max_loss:
            self.lr=self.max_loss
        elif self.lr<self.min_loss:
            self.lr=self.min_loss
    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    def __init__(self, senders, links,num):
        self.q = []
        self.aoi=[]
        self.count=0
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.packet_counter = num
        #print(self.packet_counter)
        self.queue_initial_packets()
        #print(self.packet_counter)
    def queue_initial_packets(self):

        for num,sender in enumerate(self.senders):
            sender.register_network(self)
            sender.reset_obs()
            #\print(self.packet_counter)
            noi=random.uniform(-0.5,0.5)
            if num==0:
                if Sender0_mode == 2:
                    if sender.rate == 0 or math.isnan(sender.rate):
                        print('sender.rate is 0 221')
                    heapq.heappush(self.q, (0.1 / sender.rate,0,0, sender, EVENT_TYPE_SEND, 0, 0.0,False,0, Packet(self.packet_counter,0,0,[0,1,2],"test"))) 
                if sender.rate == 0 or math.isnan(sender.rate):
                    print('sender.rate is 0 224')
                heapq.heappush(self.q, (1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,0, Packet(self.packet_counter,0,0,[0,1,2],"test"))) 
                
                
            else:
                if sender.rate == 0 or math.isnan(sender.rate):
                    print('sender.rate is 0 23 230')
                heapq.heappush(self.q, (1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,0, Packet(self.packet_counter,0.000001,0,[3,1,4],"test"))) 
            self.packet_counter+=1
    def reset(self):
        self.cur_time = 0.0
        self.q = []
        self.packet_counter
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        #print(self.packet_counter)
        self.queue_initial_packets()
    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        mes_show=True
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()
        #print(self.cur_time , end_time)
        tmpreward = np.zeros(2)
        tmp_time = self.cur_time
        total_pack_num = 0
        ca = 0
        #senders.reset_temp_and_delta_time(self.cur_time)
        #while total_pack_num < 100:
        aoi_array=[]
        while self.cur_time < end_time:
            #print("queue_delay:",self.links[1].queue_delay,"max",self.links[1].max_queue_delay,"use:",self.links[1].queue_delay/self.links[1].max_queue_delay)
            ca += 1
            #print(heapq)
            #            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 
            self.tmp_q=self.q[0]
            self.aoi.append(sender.aoi_cur_time)
            """
            if ca % 5==0:
                for link in self.links:
                   link.bw_change()
                #print(link.bw)
            """
            #input()
            #print(self.tmp_q)
            event_time,event,noise, sender, event_type, next_hop, cur_latency, dropped,acp_tmp,pack  = heapq.heappop(self.q)
            sender.B.append(sender.bytes_in_flight/8)
            sender.D.append(sender.aoi_cur_time)
            if event==1:
                #debug_pack(pack)
                if self.links[pack.path[next_hop]].link_id==0:
                    pack.link0_bw=self.links[pack.path[next_hop]].bw
                    pack.utilization0=self.links[pack.path[next_hop]].queue_delay/self.links[pack.path[next_hop]].max_queue_delay 
                elif self.links[pack.path[next_hop]].link_id==1:
                    pack.link1_bw=self.links[pack.path[next_hop]].bw   
                    pack.utilization=self.links[pack.path[next_hop]].queue_delay/self.links[pack.path[next_hop]].max_queue_delay  
                pack.now_link[0]=next_hop
                #print(pack.packet_ID,":",pack.content)
                #print("Got event %s, to link %d, latency %f at time %f" % (event_type, next_hop, cur_latency, event_time))
                #print(sender.get_aoi_reward())
                if math.isnan(event_time):
                    print('event_time')
                    #breakpoint()
                self.cur_time = event_time
                for sender_ in self.senders:
                    sender_.set_detla_time(self.cur_time) # delta time 为 curtime - aoi tmp time
                    sender_.update_temp_time(self.cur_time) # temp_time 为 cur time
                    sender_.backlog_sum += sender_.aoi_delta_time * sender_.backlog_
                    sender_.delta_sum += sender_.get_aoi_reward()
                    if REWARD_CONTROL == 1 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID] + sender_.get_aoi_reward()
                    elif REWARD_CONTROL == 2 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID] - sender_.get_aoi_reward()
                    elif REWARD_CONTROL == 3 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID]- sender_.get_aoi_reward()
                    sender_.update_cur_and_start_time() # aoi cur和start time 都增加delta time
                new_event_time = event_time
                new_event_type = event_type
                new_next_hop = pack.path_num
                #print(new_next_hop)
                new_latency = cur_latency
                new_dropped = dropped
                push_new_event = False
                #debug_pack(pack)
                #input()
                #print("flight : ",sender.get_bytes_in_flight())
                if event_type == EVENT_TYPE_ACK:
                    #print("pack.path_num == len(pack.path)-1:",pack.path_num,len(pack.path)-2)
                    if pack.path_num == len(pack.path)-1:
                        new_next_hop = pack.path_num
                        link_latency = self.links[pack.path[new_next_hop]].get_cur_latency(self.cur_time,event_type)
                        if USE_LATENCY_NOISE:
                            link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                        new_latency += link_latency
                        new_event_time += link_latency
                        new_event_type = EVENT_TYPE_RETURN
                        push_new_event = True
                        #push_new_event = EVENT_TYPE_RETURN
                        """
                            if dropped:
                                sender.on_packet_lost()
                                #print("Packet lost at time %f" % self.cur_time)
                            else:
                                sender.on_packet_acked(cur_latency)
                                #sender.set_cur_time_to_start_time()
                                sender.aoi_cur_time=event_time-pack.spawn_time
                                #print("Packet acked at time %f" % self.cur_time)
                        else:
                        """
                        if new_dropped == False:
                            new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                    else :
                        
                        new_next_hop = pack.path_num + 1
                        #print(new_next_hop)
                        #print("yo",pack.path[new_next_hop])
                        link_latency = self.links[pack.path[new_next_hop]].get_cur_latency(self.cur_time,event_type)
                        if USE_LATENCY_NOISE:
                            link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                        new_latency += link_latency
                        new_event_time += link_latency
                        push_new_event = True
                        if new_dropped == False:
                            new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                if event_type == EVENT_TYPE_RETURN:
                    if next_hop == 0:
                        if dropped:
                            sender.on_packet_lost()
                            total_pack_num +=1
                            #print("Packet lost at time %f" % self.cur_time)
                            if Sender0_mode == 1:
                                sender.cwnd/=2
                                sender.set_cwnd(sender.cwnd)
                            elif sender.sender_ID==1 and sender2_TCP:
                                sender.cwnd/=2
                                sender.set_cwnd(sender.cwnd)
                        else:
                            sender.on_packet_acked(cur_latency)
                            total_pack_num +=1
                            #sender.set_cur_time_to_start_time()
                            #if sender.sender_ID == 0:
                            #    print("Packet ack at time %f" % self.cur_time)
                            #print("Packet acked at time %f" % self.cur_time)
                            sender.link0_bw=pack.link0_bw
                            sender.link1_bw=pack.link1_bw
                            sender.utilization.append(pack.utilization)
                            sender.utilization0.append(pack.utilization0)
                            #if sender.sender_ID==0:
                            #    print("b",sender.aoi_cur_time)
                            sender.aoi_cur_time=event_time-pack.spawn_time
                            #if sender.sender_ID==0:
                            #    print("a",sender.aoi_cur_time)
                            if Sender0_mode == 2:
                                sender.acp_update(cur_latency,event_time)
                            elif Sender0_mode == 1:
                                sender.cwnd+=1
                                sender.set_cwnd(sender.cwnd)
                            elif sender.sender_ID==1 and sender2_TCP==True:
                                sender.cwnd+=1
                                sender.set_cwnd(sender.cwnd)
                        #mes_show=False
                    else:
                        new_next_hop = next_hop - 1
                        link_latency = self.links[pack.path[next_hop]].get_cur_latency(self.cur_time,event_type)
                        if USE_LATENCY_NOISE:
                            link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                        new_latency += link_latency/packet_return_size
                        new_event_time += link_latency/packet_return_size
                        push_new_event = True
                        if new_dropped == False:
                            new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                if event_type == EVENT_TYPE_SEND:
                    if next_hop == 0:      
                        sender.set_start_time()
                        if sender.can_send_packet():
                            #if sender.sender_ID == 0:
                                #print("Packet sent at time %f" % self.cur_time)
                            sender.on_packet_sent()
                            push_new_event = True
                        noi=random.uniform(-0.5,0.5)
                        if sender.sender_ID==0:
                            if sender.rate == 0 or math.isnan(sender.rate):
                                breakpoint()
                                print('sender.rate is 0 411')
                            heapq.heappush(self.q, (self.cur_time +1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,acp_tmp, Packet(self.packet_counter,self.cur_time,0,[0,1,2],"send1"))) 
                        else:
                            if sender.rate == 0 or math.isnan(sender.rate):
                                print('sender.rate is 0 415')
                            heapq.heappush(self.q, (self.cur_time +1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,acp_tmp, Packet(self.packet_counter,self.cur_time,0,[3,1,4],"send2")))  
                        self.packet_counter+=1
                    
                    else:
                        push_new_event = True
    
                    new_event_type = EVENT_TYPE_ACK
                    new_next_hop = next_hop + 1
                    
                    link_latency = self.links[pack.path[next_hop]] .get_cur_latency(self.cur_time,event_type)
                    #print(link_latency)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    if push_new_event:
                        new_latency += link_latency
                        new_event_time += link_latency
                        new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                pack.path_num=new_next_hop
                noi=random.uniform(-0.5,0.5)
                if push_new_event:
                    if math.isnan(new_event_time):
                        print('new_event_time is Nan')
                    heapq.heappush(self.q, (new_event_time,1,noi, sender, new_event_type, new_next_hop, new_latency, new_dropped,acp_tmp,pack))
                    #print(new_event_time)
                #if push_new_event and event_type==EVENT_TYPE_SEND:
                    #print("Packet sent at time %f" % self.cur_time)
                pack.now_link[1]=new_next_hop   
                #if(pack.packet_ID<10 and mes_show):
                    #print(pack.packet_ID,":",pack.content,"is",event_type,"from",pack.path[pack.now_link[0]],"to",pack.path[pack.now_link[1]])
                    #print(new_event_type)
                #print("aoi_cur_time",sender.aoi_cur_time)
                #print(self.links[1].queue_delay)
            else:
                T_=acp_tmp
               # print(self.senders[0].backlog_," ",self.senders[0].aoi_delta_time ," ", self.senders[0].backlog_sum)
                for sender_ in self.senders:
                    sender_.set_detla_time(self.cur_time)
                    sender_.update_temp_time(self.cur_time)
                    sender_.backlog_sum += sender_.aoi_delta_time * sender_.backlog_
                    sender_.delta_sum += sender_.get_aoi_reward()
                    if REWARD_CONTROL == 1 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID] + sender_.get_aoi_reward()
                    elif REWARD_CONTROL == 2 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID] - sender_.get_aoi_reward()
                    elif REWARD_CONTROL == 3 :
                        tmpreward[sender_.sender_ID] = tmpreward[sender_.sender_ID]- sender_.get_aoi_reward()
                    sender_.update_cur_and_start_time()
                #self.senders[0]
                if(T_==0):
                    self.senders[0].Bt_n=0
                    self.senders[0].Dt_n=0
                else:
                    self.senders[0].Bt_n=self.senders[0].backlog_sum/T_
                    self.senders[0].Dt_n=self.senders[0].delta_sum/T_
                self.senders[0].backlog_sum=0
                self.senders[0].delta_sum=0
                #print("D",self.senders[0].Dt_b," ",self.senders[0].Dt_n)
                #print("B",self.senders[0].Bt_b," ",self.senders[0].Bt_n)
 
                bk_tmp=self.senders[0].Bt_n-self.senders[0].Bt_b
                dk_tmp=self.senders[0].Dt_n-self.senders[0].Dt_b
                
                if bk_tmp>0 and dk_tmp>0:
                    #print("case1")
                    if self.senders[0].flag==1:
                        self.senders[0].gamma = self.senders[0].gamma + 1
                        self.senders[0].MDEC(self.senders[0].gamma)
                    else:
                        self.senders[0].DEC()
                    self.senders[0].flag=1
                elif bk_tmp>0 and dk_tmp<0:
                    #print("case2")
                    if self.senders[0].flag==1 and abs(bk_tmp) < 0.5 * abs(self.senders[0].bk_star):
                        self.senders[0].gamma = self.senders[0].gamma + 1
                        self.senders[0].MDEC(self.senders[0].gamma)  
                    else:
                        self.senders[0].INC()
                        self.senders[0].flag=0
                        self.senders[0].gamma=0
                elif bk_tmp<0 and dk_tmp>0:
                    #print("case3")
                    self.senders[0].INC()
                    self.senders[0].flag=0
                    self.senders[0].gamma=0   
                elif bk_tmp<0 and dk_tmp<0:
                    #print("case4")
                    if self.senders[0].flag==1 and self.senders[0].gamma>0:
                        self.senders[0].MDEC(self.senders[0].gamma)  
                    else:
                        self.senders[0].DEC()
                        self.senders[0].flag=0
                        self.senders[0].gamma=0
                self.senders[0].set_rate(1/self.senders[0].Z_bar+self.senders[0].bk_star/self.senders[0].T)
                #print("now:",self.senders[0].rate)
                #print("T",self.senders[0].T,"  RTT:",self.senders[0].RTT_bar,"    Z",self.senders[0].Z_bar)
                self.senders[0].Bt_b=self.senders[0].Bt_n
                self.senders[0].Dt_b=self.senders[0].Dt_n
                #print("queue: ",self.links[0].max_queue_delay)
                #for link_ in self.links:
                #    print("link",link_.link_id,": ",link_.get_cur_queue_delay(event_time,event_type))
                if math.isnan(self.senders[0].T):
                    print('self.senders[0]')
                heapq.heappush(self.q, (event_time+self.senders[0].T,0,noise, sender, event_type, next_hop, cur_latency, dropped,self.senders[0].T,pack))
        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        """
        filename=('aoi/aoi'+str(self.count)+'.json')
        self.count+=1

        with open(filename, 'w') as f:
            json.dump(self.aoi, f, indent=4)
        self.aoi=[]
        """
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt
        if REWARD_CONTROL == 0 :
            reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        elif REWARD_CONTROL == 1 :
            reward = tmpreward[0]
        elif REWARD_CONTROL == 2 :
            reward = tmpreward[0]
        elif REWARD_CONTROL == 3 :
            if self.cur_time-tmp_time==0:
                reward=0
            else:
                reward = tmpreward[0]/(self.cur_time-tmp_time)
        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))
        
        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        if self.cur_time - tmp_time == 0:
            return 0,0
        else:
            return reward * REWARD_SCALE ,tmpreward[1]/(self.cur_time-tmp_time)* REWARD_SCALE 

class Sender():
     #[Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
    def __init__(self, sender_id,rate,start,dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.sender_ID=sender_id
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.aoi_cur_time = 0
        self.aoi_delta_time = 0
        self.aoi_temp_time = 0
        self.aoi_start_time = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.net = None
        self.start=start
        #self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.link0_bw=0
        self.link1_bw=0
        self.utilization= []
        self.utilization.append(0.5)
        self.utilization0= []
        self.utilization0.append(0.5)
        self.backlog_=0
        self.delta_=0
        self.backlog_sum=0
        self.delta_sum=0
        self.acp_time=0
        self.T=0.1
        self.Bt_b=0
        self.Bt_n=0
        self.Dt_b=0
        self.Dt_n=0
        self.B=[]
        self.D=[]
        self.flag=0
        self.gamma=0
        self.Z_bar=0.1
        self.tZ=0
        self.RTT_bar=0
        self.step_size=ACP_step_size
        self.bk_star=0.25
    _next_id = 1
    def reset_temp_and_delta_time(self,curu_time):
        self.aoi_delta_time=0
        self.aoi_temp_time = curu_time
    def get_aoi_reward(self):
    
        #print(self.aoi_cur_time,self.aoi_delta_time)
        return self.aoi_cur_time*self.aoi_delta_time+self.aoi_delta_time*self.aoi_delta_time/2
    def update_temp_time(self,curu_time):
        self.aoi_temp_time = curu_time
    def set_detla_time(self,curu_time):
        if math.isnan(curu_time):
            print('curu_time')
        if math.isnan(self.aoi_temp_time):
            print('self.aoi_temp_time')
        self.aoi_delta_time = curu_time - self.aoi_temp_time
    def set_cur_time_to_start_time(self):
        if math.isnan(self.aoi_start_time):
            print('self.aoi_start_time')
        self.aoi_cur_time = self.aoi_start_time
        self.aoi_start_time=0
    def set_start_time(self):
        self.aoi_start_time=0
    def update_cur_and_start_time(self):
        if math.isnan(self.aoi_delta_time):
            print('update_cur_and_start_time')
        self.aoi_cur_time = self.aoi_cur_time + self.aoi_delta_time
        self.aoi_start_time = self.aoi_start_time + self.aoi_delta_time
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result
    def get_bytes_in_flight(self):
        return self.bytes_in_flight
    def apply_rate_delta(self, delta):
        delta *= (config.DELTA_SCALE)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))
        
    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if Sender0_mode == 1:
            #print("test")
            #print(self.bytes_in_flight,"    ",self.cwnd)
            #input()
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        elif self.sender_ID==1 and sender2_TCP == True:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET
        self.backlog_ += 1
    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)

        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.backlog_ -= 1
    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.backlog_ -= 1
        
    def set_rate(self, new_rate):
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE
        #self.rate = MAX_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        #print(smi)
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        
        obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)
        #print(self.aoi_cur_time)
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
            link0_bw=self.link0_bw,
            link1_bw=self.link1_bw,
            utilization=self.utilization,
            utilization0=self.utilization0,
            aoi=self.aoi_cur_time
        )
###############################################ACP#################
    def acp_update(self,rtt,event_time):
        self.RTT_bar=alpha*rtt+(1-alpha)*self.RTT_bar
        delta_z=event_time-self.tZ
        self.Z_bar=alpha*delta_z+(1-alpha)*self.Z_bar
        self.tZ=event_time
        self.T=min(self.RTT_bar,self.Z_bar)*10
    def INC(self):
        self.bk_star=self.step_size
        #print("inc")
    def DEC(self):
        self.bk_star=-self.step_size
        #print("dec")
    def MDEC(self,gamma):
        self.bk_star=-(1-1/(2<<gamma))*self.bytes_in_flight/8
        #print("mdec")
###############################################ACP#################       
    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()
        self.utilization = []
        self.utilization.append(0.5)
        self.utilization0 = []
        self.utilization0.append(0.5)
    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gym.Env):
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="send rate,"
                          + "recv rate,"
                          + "avg utilization,"
                          + "avg utilization0,"
                          + "Link1 bandwidth,"
                          + "Link2 bandwidth,"
                          + "aoi")):
        self.viewer = None
        self.rand = None
        self.reward_array=[]
        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.1)
        self.history_len = history_len
        #print("History length: %d" % history_len)
        self.features = features.split(",")
        #print("Features: %s" % str(self.features))
        self.packet_temp = 0
        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links,self.packet_temp)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None
        self.last_time = 0
        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)
                   

        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0
        self.reward_sum1 = 0.0
        self.reward_ewma1 = 0.0
        self.event_record = {"Events":[]}
        self.episodes_run = -1
        self.step_fir = True
        
    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        #print(sender_obs)
        return sender_obs

    def step(self, actions):
        #print("Actions: %s" % str(actions))
        #print(actions)
#        print(typeactions)
        if Sender0_mode == 0:
            action = actions
            self.senders[0].apply_rate_delta(action[0])
        elif Sender0_mode == 1:
            self.senders[0].rate = MAX_RATE
        else:
            action = actions
            self.senders[0].apply_rate_delta(action[0])
        #print(actions,' ' , self.senders[0].rate)
        #print(self.senders[0].rate)
        #self.senders[0].rate = 300 
        #self.senders[0].rate = MAX_RATE      
        now_step = math.floor(self.steps_taken /Change_rate_step)
        now_step = now_step % 2
        if now_step == 0:
            self.senders[1].rate=MAX_RATE
        else :
            self.senders[1].rate=MIN_RATE
        """
        if now_step == 0:
            self.senders[0].rate=25
        else :
            self.senders[0].rate=1
        """
        #print("Running for %fs" % self.run_dur)
        reward,reward1 = self.net.run_for_dur(self.run_dur)
        if math.isnan(reward):
            reward = 0
        if math.isnan(reward1):
            reward1 = 0
        

        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()
        #print(sender_obs)

        sender_mi = self.senders[0].get_run_data()
        sender_mi1 = self.senders[1].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        event["Reward1"] = reward1
        #event["Target Rate"] = sender_mi.target_rate
        #print( sender_mi.get("send rate"))
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        event["Link1 bandwidth"]=self.links[0].bw
        event["Link2 bandwidth"]=self.links[1].bw
        event["Send Rate1"] = sender_mi1.get("send rate")
        event["Throughput1"] = sender_mi1.get("recv rate")
        event["Utilization"]= sender_mi.get("avg utilization")
        event["Utilization0"]= sender_mi.get("avg utilization0")
        event["Aoi"]=sender_mi.get("aoi")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        #sender_obs=np.zeros(70)
        if event["Latency"] > 0.0:
            self.run_dur = 1 * sender_mi.get("avg latency")

        self.run_dur=0.5
        #print("Sender obs: %s" % sender_obs)
        sender_obs=np.array(sender_obs)
        should_stop = False
        #print("reward_sum:",self.reward_sum,"  reward:", reward)
        self.reward_sum += reward
        self.reward_sum1 += reward1
        """
        if self.steps_taken%10==0:
            print("steps_taken",self.steps_taken)
        """
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {}#,123

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()
    def update_reward_array(self,reward_):
        self.reward_array.append(reward_)
    def create_new_links_and_senders(self):
        bw    = random.uniform(self.min_bw, self.max_bw)
        lat   = random.uniform(self.min_lat, self.max_lat)
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue)))
        loss  = random.uniform(self.min_loss, self.max_loss)
        #print("queue:",queue)
        #bw    = 200
        #lat   = 0.03
        #queue = 5
        #loss  = 0.00
        self.links = [Link(0,bw, lat, queue, loss,[1,3],True,False),Link(1,bw, lat, queue, loss,[0,2,3,4],False,False),Link(2,bw, lat, queue, loss,[1,4],False,True),Link(3,bw, lat, queue, loss,[0,1],True,False),Link(4,bw, lat, queue, loss,[1,2],False,True)]
        self.senders = [Sender(0,random.uniform(0.3, 1.5) * bw, self.links[0] ,0, self.features, history_len=self.history_len),Sender(1,random.uniform(0.3, 1.5) * bw, self.links[1] ,1, self.features, history_len=self.history_len)]
        self.run_dur = 3 * lat

    def reset(self):
        self.packet_temp=self.net.packet_counter
        self.steps_taken = 0
        if step_control==0 or self.step_fir==False:
            self.step_fir = True
            self.net.reset()
            self.create_new_links_and_senders()
            self.net = Network(self.senders, self.links,self.packet_temp)
        
        else:   
            self.run_dur = 3 * random.uniform(self.min_lat, self.max_lat)
            #self.net = Network(self.senders, self.links,self.packet_temp)
        
        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file("828ver0aoidur05_log_run_%d.json" % self.episodes_run,0)
        self.event_record = {"Events":[]}
        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_ewma1 *= 0.99
        self.reward_ewma1 += 0.01 * self.reward_sum1
        print("828ver3dur05_Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        #print("527ver6dur05 _Reward1: %0.2f, Ewma Reward1: %0.2f" % (self.reward_sum1, self.reward_ewma1))
        self.reward_sum = 0.0
        self.reward_sum1 = 0.0
        self.update_reward_array(self.reward_ewma)
        self.dump_events_to_file("828ver3dur05aoi_log_reward.json",1)
        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename,index):
        if index==0:
            with open(filename, 'w') as f:
                json.dump(self.event_record, f, indent=5)
        else:
            with open(filename, 'w') as f:
                json.dump(self.reward_array, f, indent=5)
register(id='PccNs-v0', entry_point='network_sim:SimulatedNetworkEnv')
#env = SimulatedNetworkEnv()
#env.step([1.0])
