import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.05
SINGLE_DIM = 6
TOTAL_DIM = 2

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim=8, action_dim=9):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()
		self.action_dim=action_dim
		self.input_channel = 1
		self.out_channel = 1
		channel_cnn = 128
		channel_fc = 128

		self.critic_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
		self.critic_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.critic_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.critic_fc_1 = nn.Linear(self.input_channel, channel_fc)
		self.critic_fc_2 = nn.Linear(self.input_channel, channel_fc)
		self.critic_fc_3 = nn.Linear(self.input_channel, channel_fc)
		self.critic_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.critic_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4)

		self.critic_conv1.weight.data = fanin_init(self.critic_conv1.weight.data.size())
		self.critic_conv2.weight.data = fanin_init(self.critic_conv2.weight.data.size())
		self.critic_conv3.weight.data = fanin_init(self.critic_conv3.weight.data.size())
		self.critic_fc_1.weight.data = fanin_init(self.critic_fc_1.weight.data.size())
		self.critic_fc_2.weight.data = fanin_init(self.critic_fc_2.weight.data.size())
		self.critic_fc_3.weight.data = fanin_init(self.critic_fc_3.weight.data.size())
		self.critic_conv4.weight.data = fanin_init(self.critic_conv4.weight.data.size())
		self.critic_conv5.weight.data = fanin_init(self.critic_conv5.weight.data.size())

		self.fca1 = nn.Linear(self.action_dim, channel_cnn)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		# ===================Hide layer=========================
		incoming_size = 5 * channel_cnn * 5 + 3 * channel_fc + channel_fc

		self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
		# self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
		self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.out_channel)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""

		sends_batch = state[:, 0:1, :]
		throughputs_batch = state[:, 1:2, :]  ## refer to env_train.py
		latency_batch = state[:, 2:3, :]

		latency_inc_batch = state[:, 3:4, -1]
		loss_batch = state[:, 4:5, -1]
		cwnd_batch = state[:, 5:6, -1]

		avg_throughputs_batch = state[:, 6:7, :]
		avg_loss_batch = state[:, 7:8, :]

		x_1 = F.relu(self.critic_conv1(sends_batch))
		x_2 = F.relu(self.critic_conv2(throughputs_batch))
		x_3 = F.relu(self.critic_conv3(latency_batch))
		x_4 = F.relu(self.critic_fc_1(latency_inc_batch))
		x_5 = F.relu(self.critic_fc_2(loss_batch))
		x_6 = F.relu(self.critic_fc_3(cwnd_batch))
		x_7 = F.relu(self.critic_conv4(avg_throughputs_batch))
		x_8 = F.relu(self.critic_conv5(avg_loss_batch))
		a1 = F.relu(self.fca1(action))

		x_1 = x_1.view(-1, self.num_flat_features(x_1))
		x_2 = x_2.view(-1, self.num_flat_features(x_2))
		x_3 = x_3.view(-1, self.num_flat_features(x_3))
		x_4 = x_4.view(-1, self.num_flat_features(x_4))
		x_5 = x_5.view(-1, self.num_flat_features(x_5))
		x_6 = x_6.view(-1, self.num_flat_features(x_6))
		x_7 = x_7.view(-1, self.num_flat_features(x_7))
		x_8 = x_8.view(-1, self.num_flat_features(x_8))
		a1 = a1.view(-1, self.num_flat_features(a1))

		x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, a1], 1)
		x = F.relu(self.fc1(x))

		score = F.softmax(self.fc3(x), dim=1)

		return score

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


class Actor(nn.Module):

	def __init__(self, state_dim=8, action_dim=9):
		super(Actor, self).__init__()
		self.input_channel = 1
		self.action_dim = action_dim
		channel_cnn = 128
		channel_fc = 128

		self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
		self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc)
		self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc)
		self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc)
		self.actor_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4)
		self.actor_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4)

		self.actor_conv1.weight.data = fanin_init(self.actor_conv1.weight.data.size())
		self.actor_conv2.weight.data = fanin_init(self.actor_conv2.weight.data.size())
		self.actor_conv3.weight.data = fanin_init(self.actor_conv3.weight.data.size())
		self.actor_fc_1.weight.data = fanin_init(self.actor_fc_1.weight.data.size())
		self.actor_fc_2.weight.data = fanin_init(self.actor_fc_2.weight.data.size())
		self.actor_fc_3.weight.data = fanin_init(self.actor_fc_3.weight.data.size())
		self.actor_conv4.weight.data = fanin_init(self.actor_conv4.weight.data.size())
		self.actor_conv5.weight.data = fanin_init(self.actor_conv5.weight.data.size())

		# ===================Hide layer=========================
		incoming_size = 5 * channel_cnn * 5 + 3 * channel_fc  #

		self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
		# self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
		self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.action_dim)
		self.fc3.weight.data.uniform_(-EPS, EPS)

	def forward(self, state):# (发送量，吞吐量，时延，时延增量，丢包率，窗口大小, 平均吞吐量，平均丢包率)
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""

		sends_batch = state[:, 0:1, :]
		throughputs_batch = state[:, 1:2, :]  ## refer to env_train.py
		latency_batch = state[:, 2:3, :]

		latency_inc_batch = state[:, 3:4, -1]
		loss_batch = state[:, 4:5, -1]
		cwnd_batch = state[:, 5:6, -1]

		avg_throughputs_batch = state[:, 6:7, :]
		avg_loss_batch = state[:, 7:8, :]

		x_1 = F.relu(self.actor_conv1(sends_batch))
		x_2 = F.relu(self.actor_conv2(throughputs_batch))
		x_3 = F.relu(self.actor_conv3(latency_batch))
		x_4 = F.relu(self.actor_fc_1(latency_inc_batch))
		x_5 = F.relu(self.actor_fc_2(loss_batch))
		x_6 = F.relu(self.actor_fc_3(cwnd_batch))
		x_7 = F.relu(self.actor_conv4(avg_throughputs_batch))
		x_8 = F.relu(self.actor_conv5(avg_loss_batch))

		x_1 = x_1.view(-1, self.num_flat_features(x_1))
		x_2 = x_2.view(-1, self.num_flat_features(x_2))
		x_3 = x_3.view(-1, self.num_flat_features(x_3))
		x_4 = x_4.view(-1, self.num_flat_features(x_4))
		x_5 = x_5.view(-1, self.num_flat_features(x_5))
		x_6 = x_6.view(-1, self.num_flat_features(x_6))
		x_7 = x_7.view(-1, self.num_flat_features(x_7))
		x_8 = x_8.view(-1, self.num_flat_features(x_8))

		x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], 1)
		x = F.relu(self.fc1(x))

		actor = F.softmax(self.fc3(x), dim=1)

		return actor

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

